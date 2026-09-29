#!/usr/bin/env python3
"""
bag2video.py — convert  the  hackathon  .db3  LiDAR  bags  into  MP4  videos.

Views:
  pov       FIRST-PERSON  VIEW  looking  along  the  tunnel  axis
            (  forward  =  -x  )  :  perspective  pinhole  projection  from
            the  ego  origin.  The  cloud  is  2.5-D  (  x,  y  metric  ;
            the  "  z  "  slot  is  a  0..255  per-point  attribute  ,  NOT
            a  calibrated  height  )  ,  so  the  vertical  image  axis
            uses  that  attribute  scaled  by  --s-scale  (  default
            1  cm  /  unit  ->  span  0..2.55  m  )  and  the  camera  sits
            at  --cam-z  (  default  1.075  m  =  the  rail  reference  )  .
            Distance  ruler  lines  (  10/20/50/100/150  m  )  converge
            to  the  horizon  so  depth  is  readable.
  topdown   bird's-eye  (  lateral  y  across  ,  forward  -x  up  )  ,
            colored  by  the  s  attribute
  3d        orbiting  perspective  camera  around  the  scene  centroid,
            colored  by  the  s  attribute

No  ROS  is  needed  :  the  bags  use  a  non-standard  PointCloud2  CDR
(  26-byte  points  ,  data  at  byte  200  )  ,  so  they  are  read
directly  from  the  sqlite  storage  with  numpy.

Point  layout  (  26  bytes  /  point  ,  from  byte  200  of  the  blob  )  :
    [0:4]   x  float32    metres,  forward  =  -x
    [4:8]   y  float32    metres,  lateral  (  +y  =  left  assumed  )
    [8:12]  z  float32    0..255  per-point  attribute  (  "  s  "  )
    [12:16] intensity     producer  garbage  (  ignored  )
    [16:26] ring  /  ts   (  ignored  )

Usage:
    python3  bag2video.py  [  bag  ...  ]  [  --views  pov  |  topdown  |
    3d  |  all  ]  [  --fps  20  ]  [  --s-scale  0.01  ]  [  --cam-z
    1.075  ]  [  --fov  70,45  ]  [  --outdir  videos  ]

Bag  args  may  be  names  (  resolved  under  for_hackathon/  )  or  full
paths  to  a  bag  directory  containing  <name>_0.db3.
"""

import argparse
import os
import sqlite3
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 720
MAXPTS = 150_000
SEED = 42
BG = (8, 8, 10)


# ---------------------------------------------------------------- colormap
_TURBO_ANCHORS = np.array([
    [47, 14, 156], [70, 50, 215], [83, 87, 238], [91, 120, 242], [111, 163, 235],
    [144, 194, 225], [194, 217, 197], [233, 232, 137], [251, 229, 63], [249, 188, 30],
    [225, 132, 16], [188, 82, 8], [127, 25, 35], [73, 10, 17], [42, 1, 12],
], dtype=np.float64)


def turbo(t):
    t = np.clip(t, 0.0, 1.0) * (len(_TURBO_ANCHORS) - 1)
    i = np.clip(np.floor(t).astype(np.int64), 0, len(_TURBO_ANCHORS) - 2)
    f = t - i
    c = (1 - f)[:, None] * _TURBO_ANCHORS[i] + f[:, None] * _TURBO_ANCHORS[i + 1]
    return c.astype(np.uint8)


# ---------------------------------------------------------------- parsing
def parse_frame(blob):
    """26-byte  records  from  byte  200  ->  (  x,  y,  s  )  arrays."""
    raw = np.frombuffer(blob, np.uint8)[200:]
    n = len(raw) // 26
    if n == 0:
        e = np.zeros(0, np.float32)
        return e, e.copy(), e.copy()
    p = raw[:n * 26].reshape(n, 26)
    x = p[:, 0:4].view(np.float32).reshape(-1)
    y = p[:, 4:8].view(np.float32).reshape(-1)
    s = p[:, 8:12].view(np.float32).reshape(-1)
    k = (np.hypot(x, y) > 0.3) & np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    s = np.clip(np.round(s), 0.0, 255.0)
    return x[k], y[k], s[k]


def db3_path(bag):
    if os.path.isdir(bag):
        d = bag
    else:
        d = os.path.join(HERE, "for_hackathon", bag)
    names = [f for f in os.listdir(d) if f.endswith(".db3")]
    if not names:
        raise FileNotFoundError(f"no  .db3  in  {d}")
    return os.path.join(d, names[0])


def open_bag(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def iter_blobs(con):
    cur = con.execute("SELECT data FROM messages ORDER BY timestamp")
    while True:
        row = cur.fetchone()
        if row is None:
            break
        yield row[0]


def subsample_one(x, y, s, rng):
    n = len(x)
    if n > MAXPTS:
        j = rng.choice(n, MAXPTS, replace=False)
        return x[j], y[j], s[j]
    return x, y, s


# ---------------------------------------------------------------- stats
def bag_stats(path, tag):
    con = open_bag(path)
    xs, ys, ss = [], [], []
    t0 = time.time()
    n = 0
    rng = np.random.default_rng(SEED)
    for blob in iter_blobs(con):
        x, y, s = parse_frame(blob)
        if len(x) > 4000:
            j = np.arange(0, len(x), 16)
            xs.append(x[j]); ys.append(y[j]); ss.append(s[j])
        n += 1
    con.close()
    allx = np.concatenate(xs)
    ally = np.concatenate(ys)
    alls = np.concatenate(ss)
    fwd = -allx
    p98f = float(np.percentile(fwd[fwd > 0], 98)) if (fwd > 0).any() else 30.0
    st = dict(
        nmsg=n,
        p98f=p98f,
        p98y=float(np.percentile(np.abs(ally), 98)),
        smin=float(alls.min()),
        smax=float(alls.max()),
        read_s=time.time() - t0,
    )
    print(f"  [{tag}] stats:  {st['nmsg']}  msgs,  p98  forward  =  "
          f"{st['p98f']:.1f}  m,  p98  |y|  =  {st['p98y']:.1f}  m,  "
          f"s  =  [{st['smin']:.0f},  {st['smax']:.0f}]  "
          f"({st['read_s']:.0f}  s  read  )", flush=True)
    return st


# ---------------------------------------------------------------- POV
class PovParams:
    def __init__(self, s_scale=0.01, cam_z=1.075, hfov=70.0, vfov=45.0,
                 near=0.4, far=200.0):
        self.s_scale = s_scale          #  metres  per  s  unit
        self.cam_z = cam_z              #  camera  height  (  rail  reference  )
        self.near, self.far = near, far
        self.flx = (W / 2.0) / np.tan(np.deg2rad(hfov) / 2.0)
        self.fly = (H / 2.0) / np.tan(np.deg2rad(vfov) / 2.0)
        self.cx, self.cy = W / 2.0, H / 2.0


def pov_project(x, y, s, pp):
    """pinhole  looking  along  -x  ;  up  =  +s  attribute  *  s_scale."""
    f = -x
    m = (f > pp.near) & (f < pp.far)
    u = np.full(len(x), -1e9)
    v = np.full(len(x), -1e9)
    u[m] = pp.cx + pp.flx * y[m] / f[m]
    v[m] = pp.cy - pp.fly * (s[m] * pp.s_scale - pp.cam_z) / f[m]
    return u, v, f, m


def render_pov_frame(x, y, s, pp, tag, fi, nf, far_list=(10, 20, 50, 100, 150)):
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = BG
    u, v, f, m = pov_project(x, y, s, pp)
    ok = m & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if ok.any():
        ii = np.nonzero(ok)[0]
        order = np.argsort(-f[ii], kind="stable")     #  far  ->  near
        ii = ii[order]
        img[v[ii].astype(np.int64), u[ii].astype(np.int64)] = (255, 255, 255)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    fnt = ImageFont_small()
    #  distance  ruler  :  the  s  =  0  plane  at  fixed  forward  distances
    for d in far_list:
        vr = pp.cy + pp.fly * pp.cam_z / d
        if 0 <= vr < H:
            dr.line([0, int(vr), W - 1, int(vr)], fill=(45, 45, 50), width=1)
            dr.text((W - 70, int(vr) - 14), f"{d}  m", font=fnt,
                    fill=(120, 120, 125))
    dr.line([pp.cx, 0, pp.cx, H - 1], fill=(30, 30, 34), width=1)   #  y  =  0
    dr.line([0, pp.cy, W - 1, pp.cy], fill=(30, 30, 34), width=1)   #  horizon
    dr.text((20, 10),
            f"{tag}   POV  along  -x   (  s  x  {pp.s_scale*100:.1f}  cm  ,  "
            f"cam  {pp.cam_z}  m  )", font=fnt, fill=(190, 190, 195))
    dr.text((20, H - 24), f"{fi+1}/{nf}", font=fnt, fill=(150, 150, 155))
    return np.asarray(im)


# ---------------------------------------------------------------- topdown
def topdown_window(st):
    fwd = min(150.0, max(40.0, 1.15 * st["p98f"]))
    rear = 12.0
    xw = min(30.0, max(10.0, 1.25 * st["p98y"]))
    scale = min(W / (2 * xw), H / (fwd + rear))
    w_out = min(W, max(480, int(1.2 * 2 * xw * scale) // 2 * 2))
    return fwd, rear, xw, scale, w_out


def render_topdown_frame(x, y, s, st, tag, fi, nf):
    fwd, rear, xw, scale, w_out = topdown_window(st)
    img = np.zeros((H, w_out, 3), np.uint8)
    img[:, :] = BG
    px = np.rint(w_out / 2.0 + y * scale).astype(np.int64)
    py = np.rint((fwd + x) * scale).astype(np.int64)     #  forward  up,
    #                                                        ego  near  bottom
    ok = (px >= 0) & (px < w_out) & (py >= 0) & (py < H)
    if ok.any():
        t = (s[ok] - st["smin"]) / max(1e-6, st["smax"] - st["smin"])
        img[py[ok], px[ok]] = turbo(t)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    fnt = ImageFont_small()
    ey = int(fwd * scale)                                 #  ego  position
    dr.rectangle([w_out/2 - 5, ey - 5, w_out/2 + 5, ey + 5],
                 outline=(255, 255, 255))
    dr.line([w_out/2 - 10, ey, w_out/2 + 10, ey], fill=(255, 255, 255))
    bar = int(10 * scale)
    dr.line([20, H - 20, 20 + bar, H - 20], fill=(255, 255, 255), width=2)
    dr.text((20, H - 44), f"10  m", font=fnt, fill=(255, 255, 255))
    dr.text((20, 10), f"{tag}  topdown  (  forward  up  )", font=fnt,
            fill=(190, 190, 195))
    dr.text((w_out - 150, 10), f"{fi+1}/{nf}", font=fnt,
            fill=(150, 150, 155))
    return np.asarray(im)


# ---------------------------------------------------------------- 3d orbit
def render_3d(frames, st, out_path, tag, fps=20, n_sub=2, elev_deg=35.0,
              s_scale=0.01):
    allx = np.concatenate([f[0] for f in frames])
    ally = np.concatenate([f[1] for f in frames])
    target = np.array([float(np.median(allx)), float(np.median(ally)), 0.0])
    radius = min(160.0, max(45.0, 1.6 * max(st["p98f"], 30.0)))
    cel, sel = np.cos(np.deg2rad(elev_deg)), np.sin(np.deg2rad(elev_deg))
    dist = radius / cel
    fl = (H / 2.0) / np.tan(np.deg2rad(60.0) / 2.0)
    cx, cy = W / 2.0, H / 2.0

    def project(q, yaw):
        q = q - target
        cyw, syw = np.cos(yaw), np.sin(yaw)
        xr = q[:, 0] * cyw - q[:, 1] * syw
        yr = q[:, 0] * syw + q[:, 1] * cyw
        zr = q[:, 2]
        yr2 = yr * cel - zr * sel
        zc2 = yr * sel + zr * cel + dist
        keep = zc2 > 0.5
        px = np.full(q.shape[0], -1e9)
        py = np.full(q.shape[0], -1e9)
        px[keep] = cx + fl * xr[keep] / zc2[keep]
        py[keep] = cy - fl * yr2[keep] / zc2[keep]
        return px, py, zc2, keep

    total = len(frames) * n_sub
    done = 0
    ffmpeg = start_ffmpeg(out_path, fps)
    for fi, (x, y, s) in enumerate(frames):
        p = np.stack([x, y, s * s_scale], axis=1)
        t0 = (fi * n_sub) / total
        for sb in range(n_sub):
            t = (t0 + sb / n_sub) % 1.0
            img = np.zeros((H, W, 3), np.uint8)
            img[:, :] = BG
            px, py, zc, keep = project(p, 2 * np.pi * t)
            ok = keep & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            if ok.any():
                ii = np.nonzero(ok)[0]
                order = np.argsort(-zc[ii], kind="stable")
                ii = ii[order]
                tv = (s[ii] - st["smin"]) / max(1e-6, st["smax"] - st["smin"])
                img[py[ii].astype(int), px[ii].astype(int)] = turbo(tv)
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im)
            dr.text((20, 10), f"{tag}  3d  orbit  (  s  as  height  )",
                    font=ImageFont_small(), fill=(190, 190, 195))
            dr.text((W - 200, H - 24), f"{fi+1}/{len(frames)}",
                    font=ImageFont_small(), fill=(150, 150, 155))
            write_frame(ffmpeg, np.asarray(im))
            done += 1
            if done % 200 == 0:
                print(f"  [{tag}] 3d  {done}/{total}", flush=True)
    stop_ffmpeg(ffmpeg)
    print(f"  [{tag}] -> {out_path}", flush=True)


# ---------------------------------------------------------------- plumbing
def ImageFont_small(size=14):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def start_ffmpeg(out_path, fps, width=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width or W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-pix_fmt", "yuv420p", out_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def write_frame(proc, img):
    proc.stdin.write(np.ascontiguousarray(img).tobytes())


def stop_ffmpeg(proc):
    proc.stdin.close()
    proc.wait()


# ---------------------------------------------------------------- main
def process_bag(bag, views, args, outdir):
    path = db3_path(bag)
    tag = os.path.basename(os.path.normpath(bag))
    t0 = time.time()
    print(f"==  {tag}   ({path})", flush=True)
    st = bag_stats(path, tag)

    if "topdown" in views or "3d" in views:
        #  cache  subsampled  frames  (  <=  MAXPTS  )  for  the  heavy  views
        con = open_bag(path)
        frames = []
        rng = np.random.default_rng(SEED)
        nf = 0
        for blob in iter_blobs(con):
            x, y, s = parse_frame(blob)
            if len(x) < 50:
                continue
            frames.append(subsample_one(x, y, s, rng))
            nf += 1
        con.close()
        print(f"  [{tag}] {nf} frames cached", flush=True)
        if "topdown" in views:
            _, _, _, _, w_td = topdown_window(st)
            ffmpeg = start_ffmpeg(f"{outdir}/{tag}_topdown.mp4", args.fps,
                                  width=w_td)
            for fi, (x, y, s) in enumerate(frames):
                write_frame(ffmpeg, render_topdown_frame(
                    x, y, s, st, tag, fi, nf))
                if fi % 100 == 0:
                    print(f"  [{tag}] topdown  {fi+1}/{nf}", flush=True)
            stop_ffmpeg(ffmpeg)
            print(f"  [{tag}] -> {outdir}/{tag}_topdown.mp4", flush=True)
        if "3d" in views:
            render_3d(frames, st, f"{outdir}/{tag}_3d.mp4", tag,
                      fps=args.fps, s_scale=args.s_scale)

    if "pov" in views:
        #  streaming  :  one  ffmpeg,  render  on  the  fly
        pp = PovParams(s_scale=args.s_scale, cam_z=args.cam_z,
                       hfov=args.fov_h, vfov=args.fov_v)
        ffmpeg = start_ffmpeg(f"{outdir}/{tag}_pov.mp4", args.fps)
        con = open_bag(path)
        rng = np.random.default_rng(SEED)
        nf = st["nmsg"]
        fi = 0
        for blob in iter_blobs(con):
            x, y, s = parse_frame(blob)
            if len(x) < 50:
                continue
            x, y, s = subsample_one(x, y, s, rng)
            write_frame(ffmpeg, render_pov_frame(x, y, s, pp, tag, fi, nf))
            fi += 1
            if fi % 100 == 0:
                print(f"  [{tag}] pov  {fi}/{nf}", flush=True)
        con.close()
        stop_ffmpeg(ffmpeg)
        print(f"  [{tag}] -> {outdir}/{tag}_pov.mp4  ({fi}  frames  )",
              flush=True)
    print(f"  [{tag}] total  {time.time()-t0:.0f}  s\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="*",
                    help="bag names (under for_hackathon/) or paths")
    ap.add_argument("--views", default="pov",
                    choices=["pov", "topdown", "3d", "all"])
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--s-scale", type=float, default=0.01,
                    help="metres per s-unit (0.01 = 1 cm/unit, span 0-2.55 m)")
    ap.add_argument("--cam-z", type=float, default=1.075,
                    help="camera height in metres (default: rail reference)")
    ap.add_argument("--fov-h", type=float, default=70.0)
    ap.add_argument("--fov-v", type=float, default=45.0)
    ap.add_argument("--outdir", default=os.path.join(HERE, "videos"))
    a = ap.parse_args()
    views = {"pov", "topdown", "3d"} if a.views == "all" else {a.views}
    if not a.bags:
        a.bags = [os.path.join("for_hackathon", d)
                  for d in sorted(os.listdir(os.path.join(HERE, "for_hackathon")))]
    for bag in a.bags:
        process_bag(bag, views, a, a.outdir)
    print("All  done.", flush=True)


if __name__ == "__main__":
    main()
