#!/usr/bin/env python3
"""
Lidar  (theta, D)  projection review  —  black &  white  images.

For  every  point:
    r  =  sqrt(x^2 + y^2)          distance  from  the  sensor
    th =  atan2(y, -x)             angle  vs  forward  direction  (+  =  left)

For  every  D  in  0 .. 200 m  step  2 m  one  row  of  the  image  is  built:
it  is  the  VERTICAL  PLANE  PERPENDICULAR  TO  THE  TUNNEL  X  AXIS  at
distance  D  (  a  screen  standing  in  the  tunnel  )  .  Every  point  whose
axial  distance  f  =  -x  is  beyond  that  plane  (  f  >  D  )  is  projected
onto  it  along  the  ray  from  the  sensor  :  u  =  D  *  y  /  f  =  D*tan(th)
(the  plane  is  vertical  :  y  horizontal,  z  height  ;  the  2.5-D  cloud
has  no  height,  so  only  the  lateral  axis  u  is  drawn  ).
Row  D  is  a  1-D  lateral  occupancy  line;  the  whole  image  is  the  stack
of  all  D  rows  (far  at  top,  near  at  bottom).

An  obstacle  at  distance  r0  appears  in  every  row  D  <  r0  and  grows
linearly  in  apparent  width  with  D  (triangle);  just  below  row  r0  it
shows  its  true  lateral  size  —  the  cross  section  of  the  obstacle.
Summing  n = 3  consecutive  frames  fills  the  sparse  scan  gaps  and
outlines  the  shape.

No  height  exists  in  this  data  (2.5-D  cloud:  x,  y  +  0..255  slot);
only  the  horizontal  footprint  of  obstacles  is  visible.

usage:
  python3 lidar_proj.py --probe                       # stats  only
  python3 lidar_proj.py                               # all  bags,  stack  3,  per  3  frames
  python3 lidar_proj.py doubleT_platform              # one  bag
  python3 lidar_proj.py --step 3                      # one  image  per  3  frames
  python3 lidar_proj.py --nstack 3                    # stack  3  frames  per  image
  python3 lidar_proj.py --every 10                    # every  10th  frame  only
"""
import argparse
import os
import sqlite3
import glob

import cv2
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(DIR, "датасет (1)/archive/for_hackathon")

R_MAX = 200.0          # m,  rows  0 .. 200
D_STEP = 2.0           # m,  row  step
Y_HALF = 12.0          # m,  lateral  half  width  of  the  image
Y_PIX_M = 0.1          # m  per  lateral  bin
N_ROWS = int(R_MAX / D_STEP) + 1           # 101
N_COLS = int(2 * Y_HALF / Y_PIX_M) + 1     # 241


def bags():
    return sorted(glob.glob(os.path.join(DATASET, "*", "*.db3")))


def pc_data(blob):
    """(x,  y  [m],  s  [0..255])  of  the  valid  points  of  one  message"""
    raw = np.frombuffer(blob, np.uint8)[200:].reshape(-1, 26)
    x = raw[:, 0:4].view(np.float32).reshape(-1)
    y = raw[:, 4:8].view(np.float32).reshape(-1)
    s = raw[:, 8:12].view(np.float32).reshape(-1)
    k = ~((x == 0) & (y == 0) & (s == 0)) & np.isfinite(x) & np.isfinite(y)
    return x[k], y[k], s[k]


def proj_counts(x, y):
    """101 x 241  int  counts:  row  D  =  the  screen  x  =  -D,  the  vertical
    plane  perpendicular  to  the  tunnel  axis  at  distance  D  .

    A  point  with  axial  distance  f  lights  up  every  screen  D  <=  f
    (  the  screen  must  be  as  close  as  or  closer  than  the  point  ,
    otherwise  the  ray  is  stopped  by  the  point  itself  )  at  the  ray
    intersection  with  the  plane  :  u  =  D  *  y  /  f  .
    Built  as  one  big  (row,  u)  pair  scatter."""
    f = -x                                # forward  component  (positive  ahead)
    m = (f > 0.1) & (np.abs(y) < Y_HALF * 2)
    x, y, f = x[m], y[m], f[m]
    r = np.hypot(x, y)
    m2 = r <= R_MAX
    x, y, f, r = x[m2], y[m2], f[m2], r[m2]
    #  decimate  to  ~ 60 k  points  (  floor  is  over  sampled )
    step = max(1, len(r) // 60000)
    if step > 1:
        x, y, f, r = x[::step], y[::step], f[::step], r[::step]
    n = len(r)
    #  screens  a  point  contributes  to  :  D  <=  f  (  based  on  the
    #  AXIAL  distance  f,  not  on  the  range  r  :  "  beyond  the  plane
    #  at  D  "  means  the  point  stands  behind  that  plane  along  the
    #  tunnel  axis  )
    k = np.floor(f / D_STEP).astype(np.int32) + 1   # rows 0..k-1  =  D <=  f
    k = np.clip(k, 0, N_ROWS)
    kk = k > 0
    k, y, f = k[kk], y[kk], f[kk]
    P = int(k.sum())
    if P == 0:
        return np.zeros((N_ROWS, N_COLS), np.int32)
    base = np.repeat(np.cumsum(k) - k, k)
    pid = np.repeat(np.arange(len(k)), k)
    row = np.arange(P) - base                 # image  row  =  D/2  (0..k_i-1)
    u = 2.0 * row * y[pid] / f[pid]             # D =  2*row
    b = ((u + Y_HALF) / Y_PIX_M + 1e-6).astype(np.int32)   # epsilon:  bin  edges
    ok = (b >= 0) & (b < N_COLS)
    img = np.zeros((N_ROWS, N_COLS), np.int32)
    idx = row[ok] * N_COLS + b[ok]
    flat = np.bincount(idx, minlength=N_ROWS * N_COLS)
    img = flat[:N_ROWS * N_COLS].reshape(N_ROWS, N_COLS)
    return img


def render(img, title, save_path=None):
    """counts  ->  8-bit  grey  PNG  (far  up,  near  down,  left  =  -y)

    per-row  logarithmic  normalization:  the  dense  floor  saturates  to
    white,  sparser  obstacle  regions  stay  visible  against  it."""
    out = np.zeros_like(img, np.uint8)
    for i in range(N_ROWS):
        c = img[i].astype(np.float64)
        mx = c.max()
        if mx > 0:
            out[i] = np.clip(255 * np.log1p(c) / np.log1p(mx), 0, 255)
    v = 255 - out                                 # bright  points  on  black
    v = v[::-1]                                   # D = 200  at  top
    img_ = cv2.resize(v, (N_COLS * 2, N_ROWS * 16),
                      interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((img_.shape[0] + 40, img_.shape[1] + 90), np.uint8)
    canvas[40:, 90:] = img_
    # D  labels  (left)
    for i in range(0, N_ROWS, 5):
        D = (N_ROWS - 1 - i) * D_STEP
        yy = 40 + i * 16 + 8
        cv2.putText(canvas, f"{int(D)}m", (2, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 200, 1)
    # lateral  ticks  (bottom)
    for j in range(0, N_COLS, 40):
        yv = -Y_HALF + j * Y_PIX_M
        xx = 90 + j * 2
        cv2.putText(canvas, f"{yv:.0f}", (xx - 6, img_.shape[0] + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 200, 1)
    cv2.putText(canvas, title, (90, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1)
    if save_path:
        cv2.imwrite(save_path, canvas)
    return canvas


def probe(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute("SELECT data FROM messages ORDER BY timestamp").fetchall()
    d, y, s = pc_data(rows[10][0])
    r = np.hypot(d, y)
    th = np.degrees(np.arctan2(y, -d))
    print(f"==  {os.path.basename(os.path.dirname(path))}")
    print(f"  msgs:  {len(rows)}   pts/frame(med):  {len(d)}")
    print(f"  r:  [{r.min():.2f},  {np.median(r):.2f},  {r.max():.2f}]  m")
    print(f"  theta:  [{th.min():.1f},  {th.max():.1f}]  deg   (0  =  forward,  +  =  left)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="*",
                    help="bag  names  (default:  all)")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--step", type=int, default=3,
                    help="one  image  per  N  frames  (default  3)")
    ap.add_argument("--nstack", type=int, default=3,
                    help="stack  N  consecutive  frames  per  image")
    ap.add_argument("--every", type=int, default=1,
                    help="skip  frames  (1  =  all)")
    args = ap.parse_args()

    files = [p for p in bags()
             if not args.bags or any(b in p for b in args.bags)]
    for path in files:
        name = os.path.basename(os.path.dirname(path))
        if args.probe:
            probe(path)
            continue
        out_dir = os.path.join(DIR, "lidar_proj", name)
        os.makedirs(out_dir, exist_ok=True)
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        msgs = con.execute("SELECT data FROM messages ORDER BY timestamp").fetchall()
        con.close()
        n = len(msgs)
        done = 0
        for i in range(0, n, args.step * args.every):
            acc = np.zeros((N_ROWS, N_COLS), np.int32)
            cnt = 0
            for j in range(i, min(i + args.nstack, n)):
                x, y, s = pc_data(msgs[j][0])
                acc += proj_counts(x, y)
                cnt += 1
            f0 = i
            title = (f"{name}  f{f0:04d}..{f0 + cnt - 1:04d}  "
                     f"(stack {cnt})   rows = D [m],  cols = lateral [m]")
            suffix = "stack" if args.nstack > 1 else "frame"
            render(acc, title, os.path.join(out_dir, f"{suffix}_{f0:04d}.png"))
            done += 1
            if done % 50 == 0:
                print(f"  {name}:  {done}  images")
        print(f"==  {name}:  {done}  images  ->  lidar_proj/{name}/")


if __name__ == "__main__":
    main()
