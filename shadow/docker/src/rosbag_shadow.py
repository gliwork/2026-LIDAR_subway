#!/usr/bin/env python3
"""
rosbag_shadow.py  --  real-data twin of  Lidar_simulation/shadow_video.py
=========================================================================

Same  method  ,  real  sensor  input
------------------------------------
shadow_video.py  casts  every  ray  onto  a  DECISION  PLANE  at  forward
distance  D0  and  codes  the  pixel  that  the  ray  lands  on  :

      (  u  ,  v  )  =  (  lat . D0 / f ,  ver . D0 / f  )
                          0  =  open       (  no  return  in  that  ray  )
                          1  =  lit        (  nearest  return  is  BEYOND
                                            the  plane  :  background  /
                                            a  far  object  is  seen  )
                          2  =  occluded   (  nearest  return  is  IN
                                            FRONT  of  the  plane  )

A  static  "  no-object  "  scene  is  the  reference  :  a  pixel  lit  in
the  reference  but  not  lit  in  the  frame  is  an  object  SHADOW  ;
a  pixel  lit  in  the  frame  but  not  in  the  reference  is  the
object  ITSELF  (  a  new  far  return  ) .

This  script  runs  that  on  a  real  rosbag2  (  .db3  )  of
sensor_msgs/PointCloud2  .  The  only  adaptation  to  real  data  :

  *  the  reference  "  no-object  "  scene  cannot  be  re-simulated  ,
     so  it  is  estimated  from  the  data  itself  :  the  scene  is
     static  and  the  object  a  moving  minority  ,  hence  the
     temporal  CONSENSUS  (  fraction  of  frames  a  pixel  is  lit  )
     is  the  reference  .
  *  the  tunnel  axis  and  its  direction  are  AUTO-DETECTED  (  axis
     with  the  largest  span  ;  the  sign  the  cloud  extends  into  )
     --  no  hard-coded  sensor  frame  .

For  the  bundled  bag  (  cloud_with_fake_obj  ,  Hesai  "hesai_lidar"
,  ~307k  pts /  scan  @  10  Hz  )  auto-detection  gives  :

      forward  =  -y       (  tunnel  runs  into  -y  ,  ~180 m  )
      lateral  =   x       (  -3.5  ..  +12 m  ,  opens  to  the  right  )
      vertical =   z       (  floor  ~ -1.5 ,  ceiling  ~ +3.0  )

Outputs  (  under  --out  )
---------------------------
  template.png        consensus  no-object  cross-section  (  fraction  )
  frame_<i>.png       cross-section  :  lit  /  occluded  /  open  with
                      SHADOW  (  red  )  and  OBJECT  (  cyan  )  overlays
  masks/shadow_<i>.png ,  masks/object_<i>.png
  polar_<i>.png       the  same  unrolled  into  an  ( angle ,  radius  )
                      wall-ring  (  as  in  shadow_video  )
  shadow_video.mp4    the  cross-section  series
  summary.json        per-frame  shadow  /  object  /  lit  pixel  counts

Run
---
    python  rosbag_shadow.py  cloud_with_fake_obj.zst          #  .zst  input
    python  rosbag_shadow.py  .../cloud_with_fake_obj_0.db3
    python  rosbag_shadow.py  BAG  --d0  40  --res  0.1

Only  numpy  +  matplotlib  +  imageio  (+  cv2  for  masks  )  ;  no  ROS
(  the  PointCloud2  CDR  is  parsed  by  hand  )  .
"""

import argparse
import json
import math
import os
import shutil
import struct
import subprocess

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
#  PointCloud2  CDR  parsing  (  no  ROS  dependency  )
# ---------------------------------------------------------------------------

_CDR = {1: "<b", 2: "<B", 3: "<h", 4: "<H", 5: "<i", 7: "<f", 8: "<d"}


def _read_string(buf, o):
    """ROS 2  string  :  u32  len  +  bytes  +  NUL  +  pad  .  Some
    producers  count  the  NUL  in  the  length  ;  stop  at  the  first  NUL
    to  accept  both  .  The  returned  offset  is  padded  to  a  4  -
    byte  boundary  (  CDR  alignment  )  ."""
    n = struct.unpack_from("<I", buf, o)[0]
    o += 4
    o = math.ceil(o / 4) * 4                #  start  of  content
    raw = bytes(buf[o:o + n])
    nul = raw.find(b"\x00")
    if nul >= 0:                            #  NUL  counted  in  the  length
        return raw[:nul].decode("latin1"), math.ceil((o + n) / 4) * 4
    return raw.decode("latin1"), math.ceil((o + n + 1) / 4) * 4


def parse_pointcloud2(buf):
    """(  xyz  (  N , 3  )  float32  ,  intensity  (  N  )  or  None  ,  meta  )
    from  one  raw  rosbag2  CDR  message  of  sensor_msgs/PointCloud2  ."""
    o = 4                                   #  skip  CDR  encapsulation
    sec = struct.unpack_from("<i", buf, o)[0]
    nsec = struct.unpack_from("<I", buf, o + 4)[0]
    o += 8
    frame, o = _read_string(buf, o)
    height = struct.unpack_from("<I", buf, o)[0]; o += 4
    width = struct.unpack_from("<I", buf, o)[0]; o += 4
    nfields = struct.unpack_from("<I", buf, o)[0]; o += 4
    fields = []
    for _ in range(nfields):
        name, o = _read_string(buf, o)
        off = struct.unpack_from("<I", buf, o)[0]
        dt = struct.unpack_from("<B", buf, o + 4)[0]
        cnt = struct.unpack_from("<I", buf, o + 8)[0]
        o += 12
        o = math.ceil(o / 4) * 4
        fields.append((name, off, dt, cnt))
    o = math.ceil(o / 4) * 4
    o += 1                                  #  is_bigendian
    o = math.ceil(o / 4) * 4
    pstep = struct.unpack_from("<I", buf, o)[0]; o += 4
    rstep = struct.unpack_from("<I", buf, o)[0]; o += 4
    dlen = struct.unpack_from("<I", buf, o)[0]; o += 4
    o = math.ceil(o / 4) * 4
    data = bytes(buf[o:o + dlen])

    F = {nm.strip(): (off, dt) for nm, off, dt, cnt in fields}

    def col(name):
        if name not in F:
            return None
        off, dt = F[name]
        fmt = _CDR[dt]
        size = np.dtype(fmt).itemsize
        arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, pstep)
        c = arr[:, off:off + size].copy()
        return np.frombuffer(c, dtype=fmt)

    x, y, z = col("x"), col("y"), col("z")
    if x is None or y is None or z is None:
        raise ValueError("PointCloud2  has  no  x  /  y  /  z  fields  :  %s"
                         % list(F))
    intensity = col("intensity")
    n = min(len(x), len(y), len(z))
    xyz = np.stack([x[:n], y[:n], z[:n]], axis=1).astype(np.float32)
    meta = {"frame": frame, "sec": sec, "nsec": nsec, "width": width,
            "height": height, "point_step": pstep, "n_pts": int(n)}
    return xyz, (intensity[:n] if intensity is not None else None), meta


# ---------------------------------------------------------------------------
#  Bag  reading
# ---------------------------------------------------------------------------

def open_db3(path):
    """Yield  (  topic  ,  msg  bytes  ,  timestamp_ns  )  from  a  rosbag2
    .db3  .  Handles  the  MCROS  v3  layout  (  topics  +
    messages  (  topic_id  ,  timestamp  ,  data  )  )  and  the  older  layout
    (  topic  name  stored  directly  in  messages  )  ,  and  gives  a  clear
    error  on  an  empty  /  non  -  sqlite  /  wrong  -  schema  file  ."""
    import sqlite3
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise SystemExit("error  :  %s  is  empty  (  0  bytes  )  -  not  a  valid"
                         " rosbag2  .db3  .  Check  the  path  (  the  real  bag"
                         " is  usually  in  a  sub  -  folder  )  ,  or  that  the"
                         " copy  /  download  finished  ." % path)
    con = sqlite3.connect(path)
    tabs = {r[0] for r in con.execute(
        "SELECT  name  FROM  sqlite_master  WHERE  type  =  'table'  ")}
    if "messages" not in tabs:
        raise SystemExit("error  :  %s  has  no  '  messages  '  table  (  found"
                         "  :  %s  )  -  not  a  rosbag2  .db3  ."
                         % (path, sorted(tabs) if tabs else "none  -  not  a"
                            " valid  sqlite  database  "))
    cols = {r[1] for r in con.execute("PRAGMA  table_info  (  messages  )  ")}
    if "topics" in tabs and "topic_id" in cols:
        #  MCROS  v3  :  topics  (  id  ,  name  )  +  messages  (  topic_id  ,  ...  )
        topics = {tid: name for tid, name in
                  con.execute("SELECT  id  ,  name  FROM  topics  ")}
        for tid, ts, blob in con.execute(
                "SELECT  topic_id  ,  timestamp  ,  data  FROM  messages ORDER  BY  id  "):
            yield topics.get(tid, ""), bytes(blob), ts
    elif "topic" in cols:
        #  older  layout  :  topic  name  +  payload  live  in  messages
        tscol = "ts" if "ts" in cols else "timestamp"
        dcol = "data" if "data" in cols else "blob"
        for name, ts, blob in con.execute(
                "SELECT  topic  ,  %s  ,  %s  FROM  messages  ORDER  BY  id  "
                % (tscol, dcol)):
            yield name, bytes(blob), ts
    else:
        raise SystemExit("error  :  %s  has  an  unrecognised  '  messages  '"
                         " schema  (  cols  :  %s  )  ."
                         % (path, sorted(cols)))
    con.close()


def bag_path_from(zst_or_db3):
    """Return  (  db3  path  ,  tempdir-or-None  )  .  A  .zst  is  a  zstd
    -  tar  holding  one  .db3  ;  it  is  streamed  out  to  a  temp  file  .
    Symlinks  are  resolved  ;  a  missing  file  or  an  empty  extraction
    gives  a  clear  error  ."""
    import tempfile
    zst_or_db3 = os.path.realpath(os.path.expanduser(zst_or_db3))
    if not os.path.exists(zst_or_db3):
        raise SystemExit(
            "error  :  no  such  bag  (  after  resolving  links  )  :  %s"
            % zst_or_db3)
    if not zst_or_db3.endswith(".zst"):
        return zst_or_db3, None
    tmp = tempfile.mkdtemp(prefix="rosbag_shadow_")
    db3 = os.path.join(tmp, "extracted.db3")
    subprocess.run(
        ["bash", "-c",
         "zstd  -dc  '%s'  |  tar  -xf  -  --wildcards  '*.db3'  -O  >  '%s'"
         % (zst_or_db3, db3)],
        check=True)
    if not os.path.exists(db3) or os.path.getsize(db3) == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise SystemExit(
            "error  :  no  .db3  could  be  taken  out  of  %s  (  the  result  was  empty  )  .  It  must  be  a  zstd  -  tar  that  holds  a  .db3  ;  if  it  is  a  symlink  ,  its  target  has  to  live  inside  the  mounted  folder  ." % zst_or_db3)
    return db3, tmp


# ---------------------------------------------------------------------------
#  Axis  auto-detection
# ---------------------------------------------------------------------------

def detect_axes(sample_xyz):
    """(  forward_axis  ,  direction  ,  lateral_axis  ,  vertical_axis  )  as
    indices  0  =  x  ,  1  =  y  ,  2  =  z  .
    forward  =  largest  1  -  99  %  span  (  robust  to  outliers  )  ;  direction
    =  the  sign  the  cloud  extends  into  .
    lateral  vs  vertical  :  the  LATERAL  one  is  the  WIDER  of  the  two  left
    (  a  drivable  tunnel  is  wider  than  it  is  tall  )  ;  when  the  two  spans
    are  close  (  <  20  %  apart  )  ,  tie  -  break  on  symmetry  about  the
    origin  (  the  width  is  centred  on  the  sensor  ,  the  floor  /  ceiling
    stack  is  offset  )  .  This  beats  either  rule  alone  :  span  flips  on  a
    tall  /  narrow  tunnel  ,  centre  flips  when  the  sensor  sits  off  the
    width  centre  (  e.g.  a  curved  or  forked  section  )  ."""
    p01 = np.percentile(sample_xyz, 1, axis=0)
    p99 = np.percentile(sample_xyz, 99, axis=0)
    span = p99 - p01
    f = int(np.argmax(span))
    direction = -1 if np.median(sample_xyz[:, f]) < 0 else +1
    rest = [i for i in range(3) if i != f]

    def off(i):
        #  distance  of  the  1  -  99  %  range  centre  from  the  origin
        return abs((p01[i] + p99[i]) / 2.0)

    a, b = rest
    if abs(span[a] - span[b]) / max(span[a], span[b]) > 0.20:
        lateral = a if span[a] > span[b] else b     #  the  wider  one  is  lateral
    else:
        lateral = a if off(a) < off(b) else b       #  close  ->  symmetry  tie  -  break
    vertical = [i for i in rest if i != lateral][0]
    return f, direction, lateral, vertical


def forward_dist(xyz, f_axis, f_dir):
    """Forward  distance  along  the  tunnel  (  >  0  ahead  of  the
    sensor  )  =  direction  *  coordinate  ."""
    return f_dir * xyz[:, f_axis]


# ---------------------------------------------------------------------------
#  Decision-plane  coding
# ---------------------------------------------------------------------------

def grid_shape(grid, res):
    return (int(round((grid[1] - grid[0]) / res)),
            int(round((grid[3] - grid[2]) / res)))     #  (  nx  ,  nz  )


def clean_mask(mask, min_px):
    """Drop  connected  components  with  fewer  than  min_px  pixels  (  the
    beam  -  phase  flicker  of  a  real  scene  is  isolated  single  pixels  ;
    a  real  object  /  shadow  is  a  compact  cluster  )  ."""
    if min_px <= 1:
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_px
    return keep[lab]


def components(mask, grid, res, min_px=1):
    """List  of  (  size  ,  centroid_u  ,  centroid_v  )  for  components  >=
    min_px  ,  largest  first  ."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    out = []
    for k in range(1, n + 1):
        sz = int((lab == k).sum())
        if sz < min_px:
            continue
        cy, cx = ndimage.center_of_mass(lab == k)
        u = grid[0] + (cx + 0.5) * res
        v = grid[2] + (cy + 0.5) * res
        out.append((sz, float(u), float(v)))
    out.sort(reverse=True)
    return out


def auto_grid(xyz, f_axis, f_dir, lat_axis, ver_axis, D0, res, margin=0.5):
    """(  u_min  ,  u_max  ,  v_min  ,  v_max  )  from  the  projected  extent
    of  the  far  points  (  forward  >  D0  )  ."""
    f = np.maximum(forward_dist(xyz, f_axis, f_dir), 1e-6)
    u = xyz[:, lat_axis] * (D0 / f)
    v = xyz[:, ver_axis] * (D0 / f)
    far = f > D0
    if far.sum() < 50:
        far = np.ones(len(f), bool)
    u, v = u[far], v[far]
    return (float(np.percentile(u, 1) - margin),
            float(np.percentile(u, 99) + margin),
            float(np.percentile(v, 1) - margin),
            float(np.percentile(v, 99) + margin))


def project_frame(xyz, f_axis, f_dir, lat_axis, ver_axis, D0, grid, res,
                  fmin=0.3):
    """(  nz  ,  nx  )  int8  state  :  0  open  ,  1  lit  (  nearest
    beyond  plane  )  ,  2  occluded  (  nearest  in  front  )  ."""
    nx, nz = grid_shape(grid, res)
    state = np.zeros((nz, nx), np.int8)
    f = forward_dist(xyz, f_axis, f_dir)
    ok = f > fmin
    if not ok.any():
        return state
    f = f[ok]
    lat = xyz[ok, lat_axis]
    ver = xyz[ok, ver_axis]
    u = lat * (D0 / f)
    v = ver * (D0 / f)
    lo_u, hi_u, lo_v, hi_v = grid
    inb = (u >= lo_u) & (u < hi_u) & (v >= lo_v) & (v < hi_v)
    if not inb.any():
        return state
    u, v, f = u[inb], v[inb], f[inb]
    ix = np.clip(((u - lo_u) / res).astype(np.int64), 0, nx - 1)
    iz = np.clip(((v - lo_v) / res).astype(np.int64), 0, nz - 1)
    bestf = np.full((nz, nx), np.inf)
    np.minimum.at(bestf, (iz, ix), f)
    hit = bestf < np.inf
    state[hit] = np.where(bestf[hit] > D0, 1, 2)
    return state


# ---------------------------------------------------------------------------
#  Rendering
# ---------------------------------------------------------------------------

_C = {"open": (30, 30, 30), "lit": (205, 205, 205), "occ": (0, 0, 0)}
_SHADOW = (0, 0, 235)      #  red   (  BGR  )
_OBJECT = (255, 210, 0)    #  cyan  (  BGR  )


def _imshow_state(st, template, shadow, obj, lo_u, hi_u, lo_v, hi_v,
                  title):
    nx, nz = st.shape[1], st.shape[0]
    img = np.zeros((nz, nx, 3), np.uint8)
    img[st == 0] = _C["open"]
    img[st == 1] = _C["lit"]
    img[st == 2] = _C["occ"]
    if template is not None:
        img[template] = np.maximum(img[template], 80)
    if shadow is not None:
        img[shadow] = _SHADOW
    if obj is not None:
        img[obj] = _OBJECT
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img[::-1], extent=[lo_u, hi_u, lo_v, hi_v], origin="lower",
              aspect="auto", interpolation="nearest")
    ax.set_xlabel("lateral  (  m  )")
    ax.set_ylabel("vertical  (  m  )")
    ax.set_title(title, fontsize=9)
    return fig, ax


def render_state(st, template, shadow, obj, path, grid, res, title=""):
    lo_u, hi_u, lo_v, hi_v = grid
    fig, ax = _imshow_state(st, template, shadow, obj, lo_u, hi_u, lo_v,
                            hi_v, title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_template(frac, path, grid, res):
    lo_u, hi_u, lo_v, hi_v = grid
    nx, nz = grid_shape(grid, res)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(frac[::-1], extent=[lo_u, hi_u, lo_v, hi_v], origin="lower",
                   aspect="auto", cmap="gray", vmin=0, vmax=1)
    ax.set_xlabel("lateral  (  m  )")
    ax.set_ylabel("vertical  (  m  )")
    ax.set_title("consensus  no-object  reference  (  fraction  lit  )")
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def polarize(st, grid, res, Prows=96, Pcols=1024):
    """Unroll  the  (  u  ,  v  )  cross-section  into  an  ( angle ,
    radius  )  wall-ring  ,  as  in  shadow_video  .  Centre  =  grid
    centre  ;  angle  0  =  up  ,  clockwise  .  Max  -  vote  ."""
    nx, nz = grid_shape(grid, res)
    lo_u, hi_u, lo_v, hi_v = grid
    u = np.linspace(lo_u + res / 2, hi_u - res / 2, nx)
    v = np.linspace(lo_v + res / 2, hi_v - res / 2, nz)
    UU, VV = np.meshgrid(u, v)
    uc, vc = (lo_u + hi_u) / 2, (lo_v + hi_v) / 2
    du, dv = UU - uc, VV - vc
    r = np.hypot(du, dv)
    th = (np.arctan2(du, dv) + 0.5 * np.pi) % (2 * np.pi)
    rmax = max(r.max(), 1e-6)
    P = np.zeros((Prows, Pcols), np.int8)
    ii = np.clip((r / rmax * Prows).astype(np.int64), 0, Prows - 1)
    jj = np.clip((th / (2 * np.pi) * Pcols).astype(np.int64), 0, Pcols - 1)
    np.maximum.at(P, (ii, jj), st)
    return P


def render_polar(P, path, title=""):
    img = np.zeros((*P.shape, 3), np.uint8)
    img[P == 0] = (40, 40, 40)
    img[P == 1] = (225, 225, 225)
    img[P == 2] = (0, 0, 0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_xlabel("angle  around  the  tunnel  axis  (  0  =  up  )")
    ax.set_ylabel("radius")
    ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="real-data  shadow  -  gram  (  twin  of  shadow_video.py  )"
        "  from  a  rosbag2  .db3  /  .zst  of  PointCloud2  .",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help=".db3  or  a  .zst  (  zstd  -  tar  )  holding  it")
    ap.add_argument("--topic", default=None,
                    help="topic  (  default  :  the  only  one  in  the  bag  )")
    ap.add_argument("--d0", type=float, default=30.0,
                    help="decision  -  plane  forward  distance  ,  m  "
                         "(  def  30  )")
    ap.add_argument("--res", type=float, default=0.10,
                    help="cross  -  section  pixel  size  ,  m  (  def  0.1  )")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="margin  around  the  auto  grid  ,  m  (  def  0.5  )")
    ap.add_argument("--frac-thr", type=float, default=0.5,
                    help="reference  =  fraction  lit  above  this  (  def  0.5  )")
    ap.add_argument("--min-px", type=int, default=6,
                    help="drop  object  /  shadow  clusters  smaller  than  this  "
                         "(  def  6  )  -  removes  beam  flicker  noise")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0  =  all  ;  else  a  cap  (  quick  tests  )")
    ap.add_argument("--sample-frames", type=int, default=40,
                    help="frames  used  to  auto  -  detect  axes  /  grid")
    ap.add_argument("--video-step", type=int, default=2,
                    help="every  Nth  frame  in  the  video  (  def  2  )")
    ap.add_argument("--video-scale", type=int, default=8,
                    help="nearest  -  neighbour  upscale  of  the  video  "
                         "(  def  8  )")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-polar", action="store_true")
    ap.add_argument("--keep-db3", action="store_true",
                    help="keep  the  .db3  extracted  from  a  .zst  "
                         "(  default  :  delete  it  when  done  )")
    ap.add_argument("--polar-rows", type=int, default=96)
    ap.add_argument("--polar-cols", type=int, default=1024)
    ap.add_argument("--render-every", type=int, default=20,
                    help="save  frame_*.png  /  masks  every  Nth  frame")
    ap.add_argument("--out", default="rosbag_shadow_out")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)

    # ----  open  &  pick  the  topic  --------------------------------------
    db3, tmpdir = bag_path_from(a.bag)
    print("reading  %s" % db3)
    counts = {}
    for topic, _blob, _ts in open_db3(db3):
        counts[topic] = counts.get(topic, 0) + 1
    if a.topic is None:
        a.topic = max(counts, key=counts.get)
    print("topics  %s  ->  using  %r" % (counts, a.topic))

    # ----  pass  1  :  sample  frames  ->  axes  +  grid  ------------------
    sample = []
    for i, (topic, blob, _ts) in enumerate(open_db3(db3)):
        if topic != a.topic:
            continue
        if len(sample) >= a.sample_frames:
            break
        xyz, _inten, _meta = parse_pointcloud2(blob)
        sample.append(xyz)
    sample = np.vstack(sample)
    f_axis, f_dir, lat_axis, ver_axis = detect_axes(sample)
    aname = "xyz"[f_axis]
    grid = auto_grid(sample, f_axis, f_dir, lat_axis, ver_axis, a.d0,
                     a.res, a.margin)
    nx, nz = grid_shape(grid, a.res)
    print("axes  :  forward  =  %s  (  dir  %  d  )  ,  lateral  =  %s  ,  "
          "vertical  =  %s"
          % (aname, f_dir, "xyz"[lat_axis], "xyz"[ver_axis]))
    print("decision  plane  at  forward  =  %.1f  m  ;  grid  u  [%  .2f  ,  %  .2f  ]  "
          "v  [%  .2f  ,  %  .2f  ]  @  %.2f  m  ->  %dx%d  px"
          % (a.d0, grid[0], grid[1], grid[2], grid[3], a.res, nx, nz))

    # ----  pass  2  :  project  every  frame  -----------------------------
    states = []
    for i, (topic, blob, ts) in enumerate(open_db3(db3)):
        if topic != a.topic:
            continue
        if a.max_frames and i >= a.max_frames:
            break
        xyz, _inten, _meta = parse_pointcloud2(blob)
        states.append((ts, project_frame(xyz, f_axis, f_dir, lat_axis,
                                         ver_axis, a.d0, grid, a.res)))
    n = len(states)
    print("projected  %d  frames" % n)
    t0 = states[0][0]

    # ----  reference  =  consensus  (  fraction  lit  )  -------------------
    lit = np.stack([s[1] == 1 for s in states]).astype(np.float32)
    frac = lit.mean(axis=0)
    reference = frac > a.frac_thr

    # ----  per  -  frame  shadow  /  object  +  outputs  -------------------
    import cv2
    summary = {"d0_m": a.d0, "res_m": a.res, "n_frames": n,
               "forward_axis": aname, "forward_dir": f_dir,
               "lateral_axis": "xyz"[lat_axis],
               "vertical_axis": "xyz"[ver_axis],
               "grid": [float(g) for g in grid],
               "reference_frac_thr": a.frac_thr,
               "reference_lit_frac": float(reference.mean()),
               "frames": []}
    frame_imgs = []
    for i, (ts, st) in enumerate(states):
        shadow_raw = reference & (st != 1)      #  lit  in  ref  ,  not  now
        obj_raw = (st == 1) & ~reference        #  lit  now  ,  not  in  ref
        shadow = clean_mask(shadow_raw, a.min_px)
        obj = clean_mask(obj_raw, a.min_px)
        summary["frames"].append({
            "i": i, "t_ns": int(ts),
            "shadow_px": int(shadow_raw.sum()), "shadow_clean_px": int(shadow.sum()),
            "object_px": int(obj_raw.sum()), "object_clean_px": int(obj.sum()),
            "shadow_clusters": components(shadow, grid, a.res, a.min_px)[:6],
            "object_clusters": components(obj, grid, a.res, a.min_px)[:6],
            "lit_px": int((st == 1).sum()),
            "occluded_px": int((st == 2).sum())})
        if i % a.render_every == 0:
            render_state(st, reference, shadow, obj,
                         os.path.join(a.out, "frame_%04d.png" % i), grid,
                         a.res,
                         "frame  %d  (  t  =  %  .1f  s  )  :  shadow  %d  px  ,  "
                         "object  %d  px" % (i, (ts - t0) / 1e9, shadow.sum(),
                                             obj.sum()))
            cv2.imwrite(os.path.join(a.out, "masks", "shadow_%04d.png" % i),
                        (shadow.astype(np.uint8)) * 255)
            cv2.imwrite(os.path.join(a.out, "masks", "object_%04d.png" % i),
                        (obj.astype(np.uint8)) * 255)
            if not a.no_polar:
                render_polar(polarize(st, grid, a.res, a.polar_rows,
                                      a.polar_cols),
                             os.path.join(a.out, "polar_%04d.png" % i),
                             "polar  frame  %d" % i)
        if not a.no_video and i % a.video_step == 0:
            im = np.zeros((nz, nx, 3), np.uint8)
            im[st == 0] = _C["open"]
            im[st == 1] = _C["lit"]
            im[st == 2] = _C["occ"]
            im[shadow] = _SHADOW
            im[obj] = _OBJECT
            frame_imgs.append(im)

    # ----  reference  image  +  summary  +  video  -------------------------
    render_template(frac, os.path.join(a.out, "template.png"), grid, a.res)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    sh = [r["shadow_clean_px"] for r in summary["frames"]]
    ob = [r["object_clean_px"] for r in summary["frames"]]
    print("\n===  results  (  after  min  -  cluster  filter  )  ===")
    print("reference  lit  fraction  :  %  .1f  %%"
          % (100 * summary["reference_lit_frac"]))
    print("shadow  px  /  frame  :  min  %d  ,  max  %d  ,  mean  %  .0f"
          % (min(sh), max(sh), float(np.mean(sh))))
    print("object  px  /  frame  :  min  %d  ,  max  %d  ,  mean  %  .0f"
          % (min(ob), max(ob), float(np.mean(ob))))
    peak = int(np.argmax(ob))
    fr = summary["frames"][peak]
    print("strongest  object  at  frame  %d  (  t  =  %  .1f  s  )  :  %d  px  ,  "
          "top  cluster  %s"
          % (peak, (fr["t_ns"] - t0) / 1e9, fr["object_clean_px"],
             (fr["object_clusters"][:1] if fr["object_clusters"] else ["-"])))
    if not a.no_video and frame_imgs:
        vpath = os.path.join(a.out, "shadow_video.mp4")
        #  yuv420p  needs  even  dimensions  ;  scale  then  pad  if  odd
        scale = a.video_scale
        iw = nx * scale
        ih = nz * scale
        w = iw if iw % 2 == 0 else iw + 1
        h = ih if ih % 2 == 0 else ih + 1
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", "%dx%d" % (w, h), "-r", str(a.fps / a.video_step),
               "-i", "-",
               "-c:v", "libx264", "-preset", "fast", "-crf", "20",
               "-pix_fmt", "yuv420p", vpath]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for im in frame_imgs:
            if scale != 1:
                im = cv2.resize(im, (iw, ih), interpolation=cv2.INTER_NEAREST)
            if im.shape[0] != h or im.shape[1] != w:
                canvas = np.zeros((h, w, 3), np.uint8)
                canvas[:im.shape[0], :im.shape[1]] = im
                im = canvas
            proc.stdin.write(np.ascontiguousarray(im).tobytes())
        proc.stdin.close()
        proc.wait()
        print("wrote  %s  (%d  frames  @  %  .1f  fps  ,  %dx%d  )"
              % (vpath, len(frame_imgs), a.fps / a.video_step, w, h))
    if tmpdir:
        if a.keep_db3:
            print("extracted  .db3  kept  at  %s"
                  % os.path.join(tmpdir, os.path.basename(db3)))
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
            print("cleaned  up  the  temporary  .db3  extract")
    print("done  ->  %s" % a.out)


if __name__ == "__main__":
    main()
