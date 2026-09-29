#!/usr/bin/env python3
"""
lidar_slices.py —  (  y,  z  )  "  shadow  "  screens  for  a  3-D  LiDAR
point  cloud  (  .db3  rosbag2  or  .las  /  .laz  )  .

Model
-----
The  sensor  is  at  the  origin  ,  the  forward  axis  is  -x  (  use
--forward-x  if  the  file  uses  +x  )  .  For  every  slice  distance
D  =  n  ,  2n  ,  3n  ,  ...  (  --step  n  metres  )  we  take  all
points  BEYOND  the  slice  :

      f  >  D  +  gap          (  f  =  -x  ,  --gap  defaults  to  n  )

and  project  each  of  them  from  the  origin  onto  the  plane  that
is  perpendicular  to  the  axis  at  distance  D  :

      (  u  ,  v  )  =  (  y  *  D  /  f  ,  z  *  D  /  f  )

The  screen  is  accumulated  (  summed  )  into  a  (  y  ,  z  )
matrix  and  rendered  as  a  B/W  image  (  white  on  black  ,  log
scale  ,  z  up  )  .

Why  "  shadows  "
-----------------
A  real  LiDAR  sees  only  front  faces  .  A  box  at  distance  d
blocks  the  beams  behind  it  ,  so  :

  *  every  screen  with  D  +  gap  >=  d  (  BEHIND  the  box  )  loses
    the  background  returns  that  were  behind  the  box  ->  the  box
    appears  as  a  DARK  RECTANGLE  in  (  y  ,  z  )  :  its  shadow  ,
    i.e.  the  2-D  silhouette  (  width  x  height  )  magnified  by
    D  /  d  ;
  *  screens  with  D  +  gap  <  d  (  IN  FRONT  of  the  box  )  show
    the  box  itself  as  a  bright  blob  (  its  front  face  projects
    onto  the  screen  at  (  y  D  /  d  ,  z  D  /  d  )  )  .

So  a  series  of  screens  makes  each  obstacle  visible  both  as  a
blob  (  in  front  )  and  as  a  growing  dark  silhouette  (  behind
)  ;  the  shadow  grows  linearly  with  D  and  its  position  gives
(  y  ,  z  )  of  the  box  .

Inputs
------
  .las  /  .laz   :  standard  format  via  laspy  (  x  ,  y  ,  z  in
                    metres  after  the  file  scale  /  offset  )  .
  .db3  :  rosbag2  sqlite  .  Each  message  is  a  PointCloud2  :
    *  a  standard  ROS2  CDR  parse  is  tried  first  ;
    *  the  hackathon  non-standard  layout  (  26-byte  points  from
      byte  200  ,  x  /  y  /  z  float32  at  0  /  4  /  8  )  is  the
      fallback  .
  For  the  hackathon  bags  the  "  z  "  slot  is  a  0..255  per-point
  attribute  ,  not  a  height  :  pass  --z-scale  0.01  to  render  it
  as  0..2.55  m  .

Usage
-----
    python3  lidar_slices.py  FILE  [  FILE  ...  ]
        --step  5        slice  spacing  n  (  m  )
        --gap  5         exclude  the  slab  (  D  ,  D  +  gap  ]
        --start  5       first  slice  distance
        --dmax  200      last  slice  distance  (  auto  =  p99  )
        --yspan  4       lateral  half-width  (  auto  =  p99  |  y  |  )
        --zmin  -1       (  auto  =  data  range  )
        --zmax  4
        --bin  0.05      metres  per  image  cell
        --z-scale  1     multiplier  for  z  (  0.01  for  the  hackathon  )
        --forward-x      if  forward  =  +x  in  the  file
        --sheets  4      how  many  4x6  overview  sheets  (  0  =  all  )
        --outdir  slices
"""

import argparse
import os
import sqlite3
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42


# ------------------------------------------------------------------ load
def load_las(path, cap=2_000_000):
    import laspy
    L = laspy.read(path)
    x = np.asarray(L.x, np.float32)
    y = np.asarray(L.y, np.float32)
    z = np.asarray(L.z, np.float32)
    k = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[k], y[k], z[k]
    if len(x) > cap:
        j = np.random.default_rng(SEED).choice(len(x), cap, replace=False)
        x, y, z = x[j], y[j], z[j]
    return x, y, z


_DT = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
       5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def _parse_pc2_standard(blob):
    """Standard  ROS2  sensor_msgs  /  PointCloud2  CDR  (  little  -
    endian  )  ->  (  x  ,  y  ,  z  )  or  None  if  the  blob  does  not
    look  like  a  valid  standard  message  ."""
    n = len(blob)
    p = 4                                  #  CDR  version  word
    def align(a):
        nonlocal p
        p += (-p) % a
    def u32():
        nonlocal p
        if p + 4 > n:
            return None
        v = int.from_bytes(blob[p:p + 4], "little")
        p += 4
        return v
    #  frame_id  :  uint32  length  (  includes  the  NUL  )  +  string
    align(4)
    ln = u32()
    if ln is None or ln < 2 or p + ln > n:
        return None
    p += ln
    if p > n:
        return None
    align(4)
    if (height := u32()) is None or (width := u32()) is None:
        return None
    if height == 0 or width == 0 or width > 50_000_000:
        return None
    align(4)
    if (nf := u32()) is None or nf == 0 or nf > 32:
        return None
    fields = {}
    for _ in range(nf):
        align(4)
        ln = u32()
        if ln is None or ln < 2 or p + ln > n:
            return None
        name = bytes(blob[p:p + ln - 1]).decode("ascii", "ignore")
        p += ln
        align(4)
        off = u32()
        dt = blob[p]
        p += 1
        align(4)
        cnt = u32()
        sz = u32()
        if off is None or cnt is None or sz is None:
            return None
        if name in ("x", "y", "z") and dt in _DT and cnt == 1:
            fields[name] = (off, dt)
    if len(fields) < 3:
        return None
    align(1)
    if p >= n:
        return None
    be = blob[p]
    p += 1
    align(4)
    pstep = u32()
    dlen = u32()
    if be or pstep is None or dlen is None or pstep < 12 or p + dlen > n:
        return None
    if dlen % pstep != 0:
        return None
    data = np.frombuffer(blob, np.uint8, dlen, p).reshape(dlen // pstep, pstep)
    if len(data) != height * width:
        return None
    out = []
    for name in ("x", "y", "z"):
        off, dt = fields[name]
        col = data[:, off:off + np.dtype(_DT[dt]).itemsize]
        v = col.astype(np.uint8).view(_DT[dt])
        v = v.reshape(-1)
        if not np.isfinite(v.astype(np.float64)).all():
            return None
        out.append(v.astype(np.float32))
    x, y, z = out
    k = (np.abs(x) < 300) & (np.abs(y) < 300) & (np.abs(z) < 100)
    if k.mean() < 0.9:
        return None
    return x[k], y[k], z[k]


def _parse_pc2_hackathon(blob):
    """The  hackathon  layout  :  26-byte  records  from  byte  200  ;
    x  @  0  ,  y  @  4  ,  z  @  8  (  float32  )  ."""
    raw = np.frombuffer(blob, np.uint8)[200:]
    n = len(raw) // 26
    if n == 0:
        e = np.zeros(0, np.float32)
        return e, e.copy(), e.copy()
    r = raw[:n * 26].reshape(n, 26)
    x = r[:, 0:4].view(np.float32).reshape(-1)
    y = r[:, 4:8].view(np.float32).reshape(-1)
    z = r[:, 8:12].view(np.float32).reshape(-1)
    k = (np.hypot(x, y) > 0.3) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[k], y[k], z[k]


def load_db3(path, max_frames=60, per_frame=50_000):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    blobs = [r[0] for r in con.execute(
        "SELECT data FROM messages ORDER BY timestamp")]
    con.close()
    if not blobs:
        raise SystemExit(f"no  messages  in  {path}")
    step = max(1, len(blobs) // max_frames)
    blobs = blobs[::step]
    xs, ys, zs = [], [], []
    ok_std = ok_hack = 0
    for b in blobs:
        r = _parse_pc2_standard(b)
        if r is None:
            r = _parse_pc2_hackathon(b)
            ok_hack += 1
        else:
            ok_std += 1
        x, y, z = r
        if len(x) > per_frame:
            j = np.random.default_rng(SEED).choice(
                len(x), per_frame, replace=False)
            x, y, z = x[j], y[j], z[j]
        xs.append(x); ys.append(y); zs.append(z)
    print(f"  [{os.path.basename(path)}]  {len(blobs)}  frames  sampled  "
          f"(  standard  CDR  :  {ok_std}  ,  hackathon  layout  :  {ok_hack}  )  "
          f"->  {sum(len(a) for a in xs):.0f}  points")
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))


def load_any(path, max_frames=60, per_frame=50_000):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".las", ".laz"):
        return load_las(path)
    if ext == ".db3":
        return load_db3(path, max_frames, per_frame)
    raise SystemExit(f"unsupported  extension  {ext}  "
                     f"(  .las  /  .laz  /  .db3  )")


# ---------------------------------------------------------------- slices
def build_slices(f, y, z, step, gap, start, dmax,
                 yspan, zmin, zmax, bin_m):
    """For  each  D  :  sum  of  projections  of  the  points  with
    f  >  D  +  gap  onto  the  plane  at  D  .  (  f  =  forward
    distance  from  the  sensor  .  )"""
    keep = (f > 0) & (f <= dmax + gap + 1.0)
    f, y, z = f[keep], y[keep], z[keep]
    n_y = max(8, int(round(2 * yspan / bin_m)))
    n_z = max(8, int(round((zmax - zmin) / bin_m)))
    bin_y = 2.0 * yspan / n_y
    bin_z = (zmax - zmin) / n_z
    D_list = [start + i * step for i in range(int((dmax - start) / step) + 1)]
    grids = {}
    for D in D_list:
        m = f > D + gap
        u = y[m] * D / f[m]
        v = z[m] * D / f[m]
        inb = (u >= -yspan) & (u < yspan) & (v >= zmin) & (v < zmax)
        iu = ((u[inb] + yspan) / bin_y + 1e-6).astype(np.int32)
        iv = ((v[inb] - zmin) / bin_z + 1e-6).astype(np.int32)
        g = np.zeros((n_z, n_y), np.int32)
        if iu.size:
            cnt = np.bincount(iu + iv * n_y, minlength=n_z * n_y)
            g = cnt[:n_z * n_y].reshape(n_z, n_y)
        grids[D] = g
    return D_list, grids, f.size


# ---------------------------------------------------------------- render
def render(grid, D, gap, path, yspan, zmin, zmax, n_y, n_z, scale=4):
    """[n_z  ,  n_y  ]  log  -  scale  B/W  :  z  up  ,  y  left  -  right
    (  -yspan  ..  +yspan  )  ."""
    g = np.log1p(grid.astype(np.float32))
    mx = g.max()
    img = np.zeros((n_z, n_y), np.uint8)
    if mx > 0:
        img = (255 * g / mx).astype(np.uint8)
    img = img[::-1]                       #  zmax  at  the  top
    img = cv2.resize(img, (n_y * scale, n_z * scale),
                     interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, f"D  =  {D:5.1f}  m   screen  :  points  with  f  >  "
                     f"{D + gap:5.1f}  m  ,  projected  onto  the  plane  at  D",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
    cv2.putText(img, f"y:  -{yspan:.1f}  ..  +{yspan:.1f}  m  (  left  -  "
                     f"right  )  ,  z:  {zmin:.1f}  ..  {zmax:.1f}  m  "
                     f"(  bottom  -  top  )",
                (8, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                255, 1, cv2.LINE_AA)
    #  y  ticks  every  1  m
    for yy in np.arange(-yspan, yspan + 0.1, 1.0):
        px = int((yy + yspan) / (2 * yspan) * (n_y * scale))
        cv2.line(img, (px, 0), (px, 8), 150, 1)
    #  z  ticks  every  1  m
    for zz in np.arange(zmin, zmax + 0.1, 1.0):
        py = int((1 - (zz - zmin) / (zmax - zmin)) * (n_z * scale))
        cv2.line(img, (0, py), (8, py), 150, 1)
        cv2.putText(img, f"{zz:.0f}", (10, py + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 150, 1, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return img


def render_sheet(grids, D_list, tag, sheet_no, per_sheet, path,
                 yspan, zmin, zmax, n_y, n_z, cscale=1):
    cell_w = int(n_y * cscale)
    cell_h = int(n_z * cscale)
    cols, rows = 4, 6
    margin = 40
    W = cols * (cell_w + margin) + margin
    H = rows * (cell_h + margin) + margin
    sheet = np.zeros((H, W), np.uint8)
    for k in range(per_sheet):
        gi = sheet_no * per_sheet + k
        if gi >= len(D_list):
            break
        d = D_list[gi]
        g = np.log1p(grids[d].astype(np.float32))
        mx = g.max()
        cell = np.zeros((n_z, n_y), np.uint8)
        if mx > 0:
            cell = (255 * g / mx).astype(np.uint8)
        cell = cell[::-1]
        if (cell_w, cell_h) != (n_y, n_z):
            cell = cv2.resize(cell, (cell_w, cell_h),
                              interpolation=cv2.INTER_AREA)
        r, c = divmod(k, cols)
        x0 = margin + c * (cell_w + margin)
        y0 = margin + r * (cell_h + margin)
        sheet[y0:y0 + cell_h, x0:x0 + cell_w] = cell
        cv2.putText(sheet, f"D  =  {d:5.1f}  m", (x0, y0 + cell_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet)


# ----------------------------------------------------------------- main
def process(path, a):
    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(a.outdir, exist_ok=True)
    x, y, z = load_any(path, a.max_frames, a.per_frame)
    z = z * a.z_scale

    f = x if a.forward_x else -x
    fpos = f[f > 0]
    if fpos.size == 0:
        raise SystemExit("no  forward  points  (  check  --forward-x  )")
    dmax = a.dmax or min(200.0, float(np.percentile(fpos, 99)))
    start = a.start or a.step
    if start < 0.5:
        start = a.step
    if a.yspan:
        yspan = a.yspan
    else:
        yspan = min(8.0, max(3.0, float(np.percentile(np.abs(y), 99))))
    if a.zmin is not None:
        zmin = a.zmin
    else:
        zmin = float(np.percentile(z, 0.1)) - 0.2
    if a.zmax is not None:
        zmax = a.zmax
    else:
        zmax = float(np.percentile(z, 99.9)) + 0.2
    if not (zmax > zmin):
        raise SystemExit(f"bad  z  range  ({zmin}  ..  {zmax}  )")
    gap = a.gap if a.gap is not None else a.step

    print(f"  [{base}]  slices  D  =  {start}  ..  {dmax:.0f}  m  step  "
          f"{a.step}  ,  gap  {gap}  ,  y  =  ±{yspan:.2f}  m  ,  "
          f"z  =  [{zmin:.2f}  ,  {zmax:.2f}]  m  ,  bin  {a.bin_m}  m")
    D_list, grids, npts = build_slices(
        f, y, z, a.step, gap, start, dmax, yspan, zmin, zmax, a.bin_m)

    n_y = grids[D_list[0]].shape[1]
    n_z = grids[D_list[0]].shape[0]
    for D in D_list:
        p = os.path.join(a.outdir, f"{base}_D{D:05.1f}.png")
        render(grids[D], D, gap, p, yspan, zmin, zmax, n_y, n_z)
        print(f"    D  =  {D:5.1f}  m  :  {int(grids[D].sum()):7d}  pts  "
              f"->  {os.path.basename(p)}")
    nsh = a.sheets if a.sheets > 0 else max(
        1, (len(D_list) + 23) // 24)
    for s in range(nsh):
        per = 24 if s < nsh - 1 else len(D_list) - s * 24
        if per <= 0:
            break
        p = os.path.join(a.outdir, f"{base}_sheet{s:02d}.png")
        render_sheet(grids, D_list, base, s, per, p,
                     yspan, zmin, zmax, n_y, n_z, cscale=1)
        print(f"    sheet  {s}  ({per}  cells  )  ->  {os.path.basename(p)}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--step", type=float, default=5.0,
                    help="slice  spacing  n  (  m  )")
    ap.add_argument("--gap", type=float, default=None,
                    help="exclude  the  slab  (  D  ,  D  +  gap  ]  ,  default  =  step")
    ap.add_argument("--start", type=float, default=None,
                    help="first  slice  distance  (  default  =  step  )")
    ap.add_argument("--dmax", type=float, default=None,
                    help="last  slice  distance  (  default  =  p99  ,  max  200  )")
    ap.add_argument("--yspan", type=float, default=None,
                    help="lateral  half  -  width  (  m  )")
    ap.add_argument("--zmin", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)
    ap.add_argument("--bin", dest="bin_m", type=float, default=0.05,
                    help="metres  per  image  cell")
    ap.add_argument("--z-scale", type=float, default=1.0,
                    help="multiply  z  (  0.01  for  the  hackathon  0..255  attribute  )")
    ap.add_argument("--forward-x", action="store_true",
                    help="forward  =  +x  in  the  file  (  default  -x  )")
    ap.add_argument("--max-frames", type=int, default=60,
                    help=".db3  :  at  most  this  many  frames  ,  evenly  spread")
    ap.add_argument("--per-frame", type=int, default=50_000,
                    help=".db3  :  cap  points  per  sampled  frame")
    ap.add_argument("--sheets", type=int, default=0,
                    help="number  of  4x6  overview  sheets  (  0  =  all  )")
    ap.add_argument("--outdir", default=os.path.join(HERE, "slices"))
    a = ap.parse_args()
    for f in a.files:
        process(f, a)
    print("done")


if __name__ == "__main__":
    main()
