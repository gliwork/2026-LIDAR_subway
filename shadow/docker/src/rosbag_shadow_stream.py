#!/usr/bin/env python3
"""
rosbag_shadow_stream.py  --  streaming  (  single  -  pass  )  shadow  -  gram
=============================================================================

Same  decision  -  plane  method  as  rosbag_shadow.py  /  shadow_video.py  ,
but  built  for  a  bag  too  big  to  hold  in  memory  or  to  scan  twice  :

  *  the  bag  is  read  SEQUENTIALLY  (  one  sqlite  cursor  ,  in  order  )
     --  never  more  than  one  raw  point  cloud  in  memory  at  a  time  .
  *  there  is  NO  second  pass  :  the  "  no  -  object  "  reference  is  a
     SLIDING  -  WINDOW  CONSENSUS  --  for  every  new  frame  the  reference
     is  the  per  -  pixel  fraction  -  lit  over  the  last  m  frames  .
     (  a  global  consensus  over  the  whole  bag  is  not  an  option  when
     the  bag  does  not  fit  /  cannot  be  re  -  read  )  .
  *  the  output  (  video  +  per  -  frame  images  /  masks  )  is  written
     ON  THE  FLY  :  each  frame  is  encoded  straight  into  the  mp4  and
     the  PNGs  are  saved  as  they  are  computed  .

Memory
------
Only  a  ring  buffer  of  n  small  projected  state  maps  (  nz  x  nx
int8  --  a  few  kB  each  )  plus  one  raw  point  cloud  at  a  time  .
The  raw  300k  -  point  cloud  is  decoded  ,  projected  ,  and  dropped  .

Pipeline
--------
    prime  (  first  ~  prime  frames  )  ->  auto  -  detect  axes  +  grid
    then  ,  for  every  frame  :
        decode  ->  project  to  (  u  ,  v  )  state  map  ->  push  to  ring
        if  the  ring  has  >=  m  frames  :
            reference  =  fraction  -  lit  over  the  window  of  m  frames
            shadow  =  lit  in  reference  ,  not  lit  now  (  occluded  )
            object  =  lit  now  ,  not  in  reference  (  new  far  return  )
            ->  colour  image  ->  ffmpeg  (  mp4  )  +  PNG  frames  +  masks

Run
---
    python  rosbag_shadow_stream.py  cloud_with_fake_obj.zst
    python  rosbag_shadow_stream.py  BAG  --window  5  --buffer  10  --d0  30
    python  rosbag_shadow_stream.py  BAG  --excl          #  ref  =  m  frames
                                                          #  *  before  *  now

Only  numpy  +  scipy  +  cv2  ;  the  ffmpeg  CLI  encodes  the  video  .
Reuses  the  CDR  parser  /  axis  logic  of  rosbag_shadow.py  .

Show  -  all  canvas  (  --  canvas  full  ,  the  default  )  :  a  light
pre  -  scan  sizes  the  decision  -  plane  grid  to  the  FULL  far  -
point  extent  so  that  a  bend  ,  a  branch  of  a  T  -  junction  or  a
wide  cross  -  section  is  never  cropped  (  a  clipped  -  point  counter
is  shown  in  the  HUD  /  summary  )  .

Top  -  down  map  (  --  topdown  ,  on  by  default  )  :  a  cumulative
(  forward  x  lateral  )  density  map  of  the  whole  bag  .  A  straight
tunnel  is  a  vertical  band  ,  a  bend  is  a  bent  band  ,  and  a  SPLIT
/  T  -  junction  is  a  band  that  FORKS  --  so  the  complete  tunnel
topology  is  visible  in  one  image  .  Saved  as  topdown  _  map  .  png  .
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from scipy import ndimage
import cv2

try:
    import rosbag_shadow as base          #  CDR  parser  +  projection  (  same  dir  )
except ImportError:
    sys.exit("rosbag_shadow_stream.py  must  live  next  to  rosbag_shadow.py  "
             "(  it  reuses  its  PointCloud2  parser  and  projection  )  .")


# ---------------------------------------------------------------------------
#  small  helpers
# ---------------------------------------------------------------------------

def clean_mask(mask, min_px):
    """Drop  connected  components  smaller  than  min_px  (  beam  flicker  )  ."""
    if min_px <= 1:
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_px
    return keep[lab]


def top_clusters(mask, grid, res, min_px=1, k=4):
    """(  size  ,  u  ,  v  )  of  the  largest  k  components  >=  min_px  ."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    out = []
    for kk in range(1, n + 1):
        sz = int((lab == kk).sum())
        if sz < min_px:
            continue
        cy, cx = ndimage.center_of_mass(lab == kk)
        out.append((sz, float(grid[0] + (cx + 0.5) * res),
                    float(grid[2] + (cy + 0.5) * res)))
    out.sort(reverse=True)
    return out[:k]


_C = {"open": (30, 30, 30), "lit": (205, 205, 205), "occ": (0, 0, 0)}
_SHADOW = (0, 0, 235)      #  red   (  BGR  )
_OBJECT = (255, 210, 0)    #  cyan  (  BGR  )


def build_image(st, shadow, obj, scale=1):
    """(  nz  ,  nx  )  state  map  ->  BGR  image  (  high  -  z  up  ,  +  u
    right  )  ,  nearest  -  neighbour  upscaled  by  `  scale  `  ."""
    nz, nx = st.shape
    im = np.zeros((nz, nx, 3), np.uint8)
    im[st == 0] = _C["open"]
    im[st == 1] = _C["lit"]
    im[st == 2] = _C["occ"]
    im[shadow] = _SHADOW
    im[obj] = _OBJECT
    im = im[::-1]                             #  high  z  to  the  top  row
    if scale != 1:
        im = cv2.resize(im, (nx * scale, nz * scale),
                        interpolation=cv2.INTER_NEAREST)
    return im


def put_hud(im, lines, top_left=(8, 20)):
    x, y = top_left
    for ln in lines:
        cv2.putText(im, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(im, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return im


def _td_colormap(x):
    """log  -  scaled  density  ->  BGR  (  black  ->  blue  ->  cyan  ->  white  )  ."""
    x = np.clip(np.log1p(x.astype(np.float64)) /
                max(np.log1p(x.max()), 1e-6), 0, 1)
    im = np.zeros((x.shape[0], x.shape[1], 3), np.uint8)
    im[:, :, 0] = (255 * np.clip(x * 1.5, 0, 1)).astype(np.uint8)       #  B
    im[:, :, 1] = (255 * np.clip(x * 2.0 - 0.25, 0, 1)).astype(np.uint8)  #  G
    im[:, :, 2] = (255 * np.clip(x * 1.2 - 0.1, 0, 1)).astype(np.uint8)   #  R
    return im


def build_topdown_img(acc, extent, scale=4):
    """(  nfd  ,  nlat  )  count  map  +  extent  ->  BGR  (  near  =  top  row  )  ."""
    fmin, fmax, lmin, lmax = extent
    im = _td_colormap(acc)[::-1]
    if scale != 1:
        im = cv2.resize(im, (im.shape[1] * scale, im.shape[0] * scale),
                        interpolation=cv2.INTER_NEAREST)
    return im


def td_bin(lat, f, extent, res):
    """(  lat  ,  f  )  of  one  frame  ->  (  nfd  ,  nlat  )  int32  count  map  ."""
    fmin, fmax, lmin, lmax = extent
    nfd = int((fmax - fmin) / res) + 1
    nlat = int((lmax - lmin) / res) + 1
    H = np.zeros((nfd, nlat), np.int32)
    m = (f >= fmin) & (f <= fmax) & (lat >= lmin) & (lat <= lmax)
    if m.any():
        ii = np.clip(((f[m] - fmin) / res).astype(int), 0, nfd - 1)
        jj = np.clip(((lat[m] - lmin) / res).astype(int), 0, nlat - 1)
        np.add.at(H, (ii, jj), 1)
    return H


def start_ffmpeg(path, w, h, fps):
    h2 = h if h % 2 == 0 else h + 1
    w2 = w if w % 2 == 0 else w + 1
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", "%dx%d" % (w2, h2), "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    return proc, (w2, h2)


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="streaming  (  single  -  pass  )  shadow  -  gram  from  a  "
                    "rosbag2  .db3  /  .zst  of  PointCloud2  .")
    ap.add_argument("bag", help=".db3  or  a  .zst  (  zstd  -  tar  )  holding  it")
    ap.add_argument("--topic", default=None,
                    help="topic  (  default  :  the  first  one  seen  )")
    ap.add_argument("--d0", type=float, default=30.0,
                    help="decision  -  plane  forward  distance  ,  m")
    ap.add_argument("--res", type=float, default=0.10,
                    help="cross  -  section  pixel  size  ,  m")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="margin  around  the  grid  ,  m")
    ap.add_argument("--canvas", choices=["full", "auto"], default="full",
                    help="'full'  :  pre  -  scan  the  bag  for  the  full  far  -  "
                         "point  (  u  ,  v  )  extent  so  a  bend  /  T  -  branch  /  "
                         "wide  section  is  never  cropped  ;  'auto'  :  size  the  "
                         "grid  from  the  priming  frames  only  (  old  behaviour  )  .")
    ap.add_argument("--canvas-sample", type=int, default=1,
                    help="extent  pre  -  scan  uses  every  Nth  frame  (  1  =  all  )")
    ap.add_argument("--canvas-pct", type=float, default=0.0,
                    help="canvas  extent  percentile  :  0  =  true  min  /  max  "
                         "(  show  -  all  ,  the  default  )  ;  e.g.  0.5  =  the  "
                         "0.5  /  99.5  percentile  (  robust  to  a  lone  outlier  )")
    ap.add_argument("--topdown", action="store_true", default=True,
                    help="build  the  cumulative  top  -  down  tunnel  map  "
                         "(  on  by  default  )")
    ap.add_argument("--no-topdown", dest="topdown", action="store_false",
                    help="skip  the  top  -  down  map  /  video ")
    ap.add_argument("--td-res", type=float, default=0.5,
                    help="top  -  down  map  pixel  size  ,  m  (  default  0.5  )")
    ap.add_argument("--td-video", default="", metavar="PATH",
                    help="per  -  frame  top  -  down  video  (  ''  =  skip  ,  "
                         "the  default  )")
    ap.add_argument("--td-video-scale", type=int, default=4,
                    help="upscale  of  the  top  -  down  video  (  default  4  )")
    ap.add_argument("--prime", type=int, default=5,
                    help="frames  to  read  first  in  order  to  auto  -  "
                         "detect  axes  +  grid")
    ap.add_argument("--buffer", type=int, default=10, metavar="N",
                    help="ring  -  buffer  of  projected  frames  to  keep  "
                         "(  N  ,  default  10  )")
    ap.add_argument("--window", type=int, default=5, metavar="M",
                    help="sliding  -  window  of  M  frames  used  for  the  "
                         "local  reference  (  M  ,  default  5  )")
    ap.add_argument("--excl", action="store_true",
                    help="build  the  reference  from  the  M  frames  "
                         "BEFORE  the  current  one  (  default  :  the  window  "
                         "includes  the  current  frame  )")
    ap.add_argument("--frac-thr", type=float, default=0.5,
                    help="reference  =  fraction  lit  above  this")
    ap.add_argument("--min-px", type=int, default=6,
                    help="drop  object  /  shadow  clusters  smaller  than  this")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0  =  all  ;  else  a  cap")
    ap.add_argument("--video", default="shadow_stream.mp4",
                    help="output  video  (  ''  to  skip  )")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--video-step", type=int, default=1,
                    help="put  every  Nth  frame  in  the  video")
    ap.add_argument("--video-scale", type=int, default=8,
                    help="nearest  -  neighbour  upscale  of  the  video")
    ap.add_argument("--frames-dir", default="",
                    help="dir  for  per  -  frame  PNGs  (  default  :  <  out  >  /  frames  )")
    ap.add_argument("--masks-dir", default="",
                    help="dir  for  shadow  /  object  masks  (  default  :  <  out  >  /  masks  )")
    ap.add_argument("--no-frames", action="store_true",
                    help="do  not  write  the  per  -  frame  PNGs")
    ap.add_argument("--no-masks", action="store_true",
                    help="do  not  write  the  shadow  /  object  masks")
    ap.add_argument("--save-window", default="",
                    help="dir  to  dump  the  sliding  -  window  reference  "
                         "(  ''  to  skip  )")
    ap.add_argument("--render-every", type=int, default=1,
                    help="write  PNG  frames  every  Nth  output  frame")
    ap.add_argument("--keep-db3", action="store_true",
                    help="keep  the  .db3  extracted  from  a  .zst")
    ap.add_argument("--out", default="cwf_shadow_stream",
                    help="base  output  dir  (  for  summary  +  defaults  )")
    a = ap.parse_args(argv)

    if a.window > a.buffer:
        a.buffer = a.window            #  the  buffer  must  hold  the  window
    os.makedirs(a.out, exist_ok=True)
    frames_dir = a.frames_dir or os.path.join(a.out, "frames")
    masks_dir = a.masks_dir or os.path.join(a.out, "masks")
    win_dir = a.save_window
    if not a.no_frames:
        os.makedirs(frames_dir, exist_ok=True)
    if not a.no_masks:
        os.makedirs(masks_dir, exist_ok=True)
    if win_dir:
        os.makedirs(win_dir, exist_ok=True)

    # ----  open  the  bag  (  single  sequential  cursor  )  ---------------
    db3, tmpdir = base.bag_path_from(a.bag)
    print("reading  (  streaming  )  %s" % db3)
    gen = base.open_db3(db3)

    t0ns = None
    grid = None
    f_axis = f_dir = lat = ver = None
    aname = None
    prime = []
    ring = []                       #  [(  ts  ,  state  )  ]  ,  oldest  first
    ffmpeg = None
    vw = vh = None
    t0_wall = None
    n_out = 0
    n_read = 0
    summary_frames = []
    extent = None                      #  (  fmin  ,  fmax  ,  lmin  ,  lmax  )  top  -  down
    td_map = None                      #  (  nfd  ,  nlat  )  per  -  frame  top  -  down
    td_acc = None                      #  (  nfd  ,  nlat  )  cumulative  (  whole  bag  )
    td_ffmpeg = td_vw = td_vh = None
    td_res = a.td_res
    clip_far_max = 0

    def emit(ts, st):
        """Build  the  reference  from  the  window  and  write  everything
        for  the  current  (  latest  )  frame  ."""
        nonlocal ffmpeg, vw, vh, n_out, t0_wall, td_ffmpeg, td_vw, td_vh, td_map, td_acc, clip_far_max
        need = a.window if not a.excl else a.window + 1
        if len(ring) < need:
            return
        if a.excl:
            win = [s for _, s in ring[-(a.window + 1):-1]]
        else:
            win = [s for _, s in ring[-a.window:]]
        cur = ring[-1][1]
        lit = np.stack([w == 1 for w in win]).astype(np.float32)
        ref = lit.mean(axis=0) > a.frac_thr

        shadow = clean_mask(ref & (cur != 1), a.min_px)
        obj = clean_mask((cur == 1) & ~ref, a.min_px)

        if t0_wall is None:
            t0_wall = ts
        dt = (ts - t0_wall) / 1e9

        # ----  video  (  on  the  fly  )  --------------------------------
        #  diagnostic  :  far  points  OUTSIDE  the  canvas  (  0  =  show  -  all  OK  )
        farr = base.forward_dist(xyz, f_axis, f_dir)
        fm = farr > a.d0
        if fm.any():
            uu = xyz[fm, lat] * a.d0 / farr[fm]
            vv = xyz[fm, ver] * a.d0 / farr[fm]
            clip_far_max = max(clip_far_max, int(
                ((uu < grid[0]) | (uu > grid[1]) |
                 (vv < grid[2]) | (vv > grid[3])).sum()))
        #  ----  top  -  down  map  (  the  tunnel  shape  /  fork  )  --------
        if a.topdown and extent is not None:
            td_map = td_bin(xyz[:, lat], farr, extent, td_res)
            if td_acc is not None:
                td_acc += td_map
            if a.td_video and (n_out % a.video_step == 0):
                tim = build_topdown_img(td_map, extent, a.td_video_scale)
                tim = put_hud(tim, ["t=%6.1fs" % dt,
                                    "fwd 0..%.0f m  lat %.0f..%.0f m"
                                    % (extent[1], extent[2], extent[3])], (8, 44))
                if td_ffmpeg is None:
                    td_ffmpeg, (td_vw, td_vh) = start_ffmpeg(
                        os.path.join(a.out, a.td_video),
                        tim.shape[1], tim.shape[0], a.fps / a.video_step)
                if tim.shape[0] != td_vh or tim.shape[1] != td_vw:
                    c2 = np.zeros((td_vh, td_vw, 3), np.uint8)
                    c2[:tim.shape[0], :tim.shape[1]] = tim
                    tim = c2
                td_ffmpeg.stdin.write(np.ascontiguousarray(tim).tobytes())
        if a.video and (n_out % a.video_step == 0):
            im = build_image(cur, shadow, obj, a.video_scale)
            im = put_hud(im, [
                "t=%6.1fs  f=%d" % (dt, n_out),
                "shadow %d  object %d  clip %d"
                % (int(shadow.sum()), int(obj.sum()), clip_far_max)])
            if ffmpeg is None:
                ffmpeg, (vw, vh) = start_ffmpeg(
                    os.path.join(a.out, a.video),
                    im.shape[1], im.shape[0], a.fps / a.video_step)
            if im.shape[0] != vh or im.shape[1] != vw:
                canvas = np.zeros((vh, vw, 3), np.uint8)
                canvas[:im.shape[0], :im.shape[1]] = im
                im = canvas
            ffmpeg.stdin.write(np.ascontiguousarray(im).tobytes())

        # ----  per  -  frame  PNGs  +  masks  (  on  the  fly  )  ---------
        if n_out % a.render_every == 0:
            if not a.no_frames:
                p = os.path.join(frames_dir, "frame_%05d.png" % n_out)
                big = build_image(cur, shadow, obj, max(a.video_scale, 6))
                big = put_hud(big, [
                    "f=%d  t=%5.1fs" % (n_out, dt),
                    "shadow %d  object %d  clip %d"
                    % (int(shadow.sum()), int(obj.sum()), clip_far_max)])
                cv2.imwrite(p, big)
            if a.topdown and extent is not None and td_map is not None:
                cim = build_topdown_img(td_map, extent, max(6, a.td_video_scale))
                cim = put_hud(cim, ["topdown  f=%d" % n_out], (8, 44))
                cv2.imwrite(os.path.join(frames_dir, "topdown_%05d.png" % n_out),
                            cim)
            if not a.no_masks:
                cv2.imwrite(os.path.join(masks_dir, "shadow_%05d.png" % n_out),
                            (shadow.astype(np.uint8)) * 255)
                cv2.imwrite(os.path.join(masks_dir, "object_%05d.png" % n_out),
                            (obj.astype(np.uint8)) * 255)
            if win_dir:
                refimg = (ref.astype(np.float32) * 255).astype(np.uint8)
                refimg = refimg[::-1]
                refimg = cv2.resize(refimg, (refimg.shape[1] * 6,
                                             refimg.shape[0] * 6),
                                    interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(win_dir, "window_%05d.png" % n_out),
                            refimg)

        # ----  summary  record  (  compact  )  --------------------------
        summary_frames.append({
            "i": n_out, "t_ns": int(ts),
            "shadow_px": int(shadow.sum()), "object_px": int(obj.sum()),
            "object_clusters": top_clusters(obj, grid, a.res, a.min_px, 3),
            "shadow_clusters": top_clusters(shadow, grid, a.res, a.min_px, 3)})

        n_out += 1
        if n_out % 50 == 0:
            print("  ...  %d  frames  out  (  t  =  %  .1f  s  )"
                  % (n_out, dt), flush=True)

    # ----  the  sequential  loop  -----------------------------------------
    #  ----  show  -  all  canvas  :  pre  -  scan  the  full  (  u  ,  v  )  extent  --
    if a.canvas == "full":
        um = vm = -np.inf
        Un = Vm = np.inf
        fmAx = 0.0
        la_min = np.inf
        la_max = -np.inf
        axs = None
        ax_sample = None
        nscan = 0
        print("pre  -  scan  for  the  show  -  all  canvas  "
              "(  every  %dth  frame  )  ..." % max(a.canvas_sample, 1), flush=True)
        for t2, b2, _ts2 in gen:
            if a.topic is None:
                a.topic = t2
            if t2 != a.topic:
                continue
            nscan += 1
            if a.max_frames and nscan > a.max_frames:
                break
            if (nscan - 1) % a.canvas_sample != 0:
                continue
            xyzs, _i2, _m2 = base.parse_pointcloud2(b2)
            if axs is None:                    #  detect  axes  once  ,  reuse
                axs = base.detect_axes(xyzs)   #  (  f_axis  ,  f_dir  ,  lat  ,  ver  )
                ax_sample = xyzs
            f_ax, f_d, lat_ax, ver_ax = axs
            f2 = base.forward_dist(xyzs, f_ax, f_d)
            fm2 = np.isfinite(f2) & (f2 > a.d0)
            if fm2.any():
                fmAx = max(fmAx, float(f2[fm2].max()))
                u2 = xyzs[fm2, lat_ax] * a.d0 / f2[fm2]
                v2 = xyzs[fm2, ver_ax] * a.d0 / f2[fm2]
                g2 = np.isfinite(u2) & np.isfinite(v2)
                if g2.any():
                    u2, v2 = u2[g2], v2[g2]
                    if a.canvas_pct <= 0:      #  show  -  all  (  min  /  max  )
                        um = max(um, float(u2.max())); Un = min(Un, float(u2.min()))
                        vm = max(vm, float(v2.max())); Vm = min(Vm, float(v2.min()))
                    else:                      #  percentile  extent
                        p = a.canvas_pct
                        um = max(um, float(np.percentile(u2, 100 - p)))
                        Un = min(Un, float(np.percentile(u2, p)))
                        vm = max(vm, float(np.percentile(v2, 100 - p)))
                        Vm = min(Vm, float(np.percentile(v2, p)))
            ll = xyzs[:, lat_ax]
            ll = ll[np.isfinite(ll)]
            if ll.size:
                la_min = min(la_min, float(ll.min()))
                la_max = max(la_max, float(ll.max()))
        if np.isfinite(um) and ax_sample is not None:
            f_axis, f_dir, lat, ver = base.detect_axes(ax_sample)
            aname = "xyz"[f_axis]
            grid = (Un - a.margin, um + a.margin,
                    Vm - a.margin, vm + a.margin)
            extent = (0.0, fmAx + a.margin, la_min - a.margin, la_max + a.margin)
            if a.topdown:
                td_acc = np.zeros((int((extent[1] - extent[0]) / td_res) + 1,
                                   int((extent[3] - extent[2]) / td_res) + 1),
                                  np.int32)
            print("  show  -  all  canvas  :  fmax  %.0f  m  ;  lateral  %.1f..%.1f  m  ;  "
                  "grid  %dx%d  px  @  %.2f  m"
                  % (fmAx, la_min, la_max,
                     *base.grid_shape(grid, a.res), a.res), flush=True)
        print("  (  %d  frames  scanned  )" % nscan, flush=True)
        gen = base.open_db3(db3)        #  the  pre  -  scan  consumed  the  cursor  ;
                                        #  re  -  open  for  the  main  pass

    for topic, blob, ts in gen:
        if a.topic is None:
            a.topic = topic            #  lock  onto  the  first  topic  seen
        if topic != a.topic:
            continue
        n_read += 1
        if a.max_frames and n_read > a.max_frames:
            break
        xyz, _inten, _meta = base.parse_pointcloud2(blob)

        if grid is None and a.canvas == "auto":
            prime.append(xyz)
            if len(prime) < a.prime:
                continue                    #  keep  priming  ...
            sample = np.vstack(prime)
            f_axis, f_dir, lat, ver = base.detect_axes(sample)
            aname = "xyz"[f_axis]
            grid = base.auto_grid(sample, f_axis, f_dir, lat, ver, a.d0,
                                  a.res, a.margin)
            for p in prime:
                ring.append((None, base.project_frame(
                    p, f_axis, f_dir, lat, ver, a.d0, grid, a.res)))
            prime = []
            print("axes  :  forward  =  %s  (  dir  %  d  )  ,  lateral  =  %s  ,  "
                  "vertical  =  %s  ;  grid  %dx%d  px  @  %.2f  m"
                  % (aname, f_dir, "xyz"[lat], "xyz"[ver],
                     *base.grid_shape(grid, a.res), a.res), flush=True)
            emit(ts, ring[-1][1])           #  the  last  primed  =  current
            continue

        st = base.project_frame(xyz, f_axis, f_dir, lat, ver, a.d0, grid,
                                a.res)
        ring.append((ts, st))
        if len(ring) > a.buffer:
            ring.pop(0)
        emit(ts, st)
        if n_read % 100 == 0:
            print("  ...  read  %d  frames" % n_read, flush=True)

    # ----  wrap  -  up  ----------------------------------------------------
    if ffmpeg is not None:
        ffmpeg.stdin.close()
        ffmpeg.wait()
        print("wrote  %s  (%d  video  frames  )"
              % (os.path.join(a.out, a.video), n_out), flush=True)
    if td_ffmpeg is not None:
        td_ffmpeg.stdin.close()
        td_ffmpeg.wait()
        print("wrote  %s" % os.path.join(a.out, a.td_video), flush=True)

    #  ----  the  cumulative  top  -  down  map  (  whole  tunnel  shape  )  -------
    if a.topdown and td_acc is not None and extent is not None:
        fmin, fmax, lmin, lmax = extent
        big = _td_colormap(td_acc)[::-1]
        big = cv2.resize(big, (big.shape[1] * 4, big.shape[0] * 4),
                         interpolation=cv2.INTER_NEAREST)
        big = put_hud(big, [
            "topdown  tunnel  map  (  whole  bag  )",
            "y  :  fwd  0..%.0f  m   (  up  =  near  )" % fmax,
            "x  :  lateral  %.1f..%.1f  m" % (lmin, lmax)], (10, 26))
        mp = os.path.join(a.out, "topdown_map.png")
        cv2.imwrite(mp, big)
        print("wrote  %s  (  full  tunnel  top  -  down  ,  %dx%d  )"
              % (mp, big.shape[1], big.shape[0]), flush=True)

    summary = {"d0_m": a.d0, "res_m": a.res, "n_read": n_read,
               "n_output": n_out, "window": a.window,
               "window_excludes_current": bool(a.excl),
               "canvas": a.canvas, "canvas_pct": a.canvas_pct,
               "clip_far_max": int(clip_far_max),
               "topdown_extent": ([float(e) for e in extent] if extent else None),
               "forward_axis": aname, "forward_dir": int(f_dir) if f_dir is not None else None,
               "lateral_axis": ("xyz"[lat] if lat is not None else None),
               "vertical_axis": ("xyz"[ver] if ver is not None else None),
               "grid": [float(g) for g in grid] if grid is not None else None,
               "frames": summary_frames}
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    if summary_frames:
        sh = [r["shadow_px"] for r in summary_frames]
        ob = [r["object_px"] for r in summary_frames]
        print("\n===  streaming  results  (  %d  frames  in  ,  %d  out  )  ==="
              % (n_read, n_out))
        print("shadow  px  /  frame  :  min  %d  ,  max  %d  ,  mean  %  .0f"
              % (min(sh), max(sh), float(np.mean(sh))))
        print("object  px  /  frame  :  min  %d  ,  max  %d  ,  mean  %  .0f"
              % (min(ob), max(ob), float(np.mean(ob))))
        print("far  points  OUTSIDE  the  canvas  (  clipped  )  :  max  %d  /  frame  "
              "(  0  =  show  -  all  OK  )" % clip_far_max)

    if tmpdir and not a.keep_db3:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("cleaned  up  the  temporary  .db3  extract")
    elif tmpdir:
        print("extracted  .db3  kept  at  %s"
              % os.path.join(tmpdir, os.path.basename(db3)))
    print("done  ->  %s" % a.out)


if __name__ == "__main__":
    main()
