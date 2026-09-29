#!/usr/bin/env python3
"""
Tunnel  cross  sections  at  a  fixed  distance  D   (black  &  white).

For  every  chosen  D  (2 .. 180 m)  the  cross  section  is  taken  on  the
VERTICAL  PLANE  PERPENDICULAR  TO  THE  TUNNEL  X  AXIS  at  distance  D:
the  slab  of  points  whose  axial  coordinate  f  =  -x  lies  in
[D  -  slice,  D]  (  a  5  cm  physical  slab  between  two  parallel  planes  )
is  projected  ONTO  THAT  PLANE  ORTHOGONALLY  —  a  point  of  the  slab  is
already  on  the  plane,  so  it  keeps  its  own  lateral  coordinate  :

    u  =  y        (  no  angular  re-projection  :  the  walls  stay  at
                   their  true  lateral  position  ,  no  1/cos  stretch  )

(the  plane  itself  is  vertical  :  y  horizontal,  z  height.  The  cloud
is  2.5-D  with  no  height  channel,  so  only  the  lateral  axis  is  drawn.)

The  result  is  a  1-D  lateral  profile  —  the  true  shape  of  the  tunnel
on  that  plane:

  *  the  solid  band  =  the  corridor  floor  between  the  walls
  *  the  left /  right  edges  of  the  band  =  the  tunnel  walls
  *  a  bump  outside  the  band  or  a  dense  clump  inside  =  an  obstacle

Limitations:  the  cloud  is  2.5-D  (x,  y  +  0..255  intensity  slot,  no
height).  The  section  is  lateral  only;  "  on  rail  /  on  gauge  /
hanging  from  the  ceiling  "  cannot  be  separated  here.  The  sensor  is
1075  mm  above  the  rail  heads  —  remember  that  when  reading  the
lateral  edges.

Outputs  (  per  bag,  in  tunnel_section/  ):
  sheet_<bag>.png            all  D  sections  stacked  (  contact  sheet  )
  width_<bag>.png            tunnel  width  vs  D  (  median  over  frames  )
  width_<bag>.csv            per  (  frame,  D  ):  left /  right  edge  [m],
                             width  [m],  centre  offset  [m]

usage:
  python3 tunnel_section.py                                  # all  bags
  python3 tunnel_section.py doubleT_platform                 # one  bag
  python3 tunnel_section.py --D 10 20 40 --nstack 9          # custom  D  list
"""
import argparse
import csv
import glob
import os
import sqlite3

import cv2
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(DIR, "датасет (1)/archive/for_hackathon")

Y_HALF = 12.0            # m,  half  width  of  the  section
BIN_M = 0.05             # m  per  lateral  bin
N_BINS = int(2 * Y_HALF / BIN_M) + 1     # 481
R_BEYOND_MAX = 200.0     # ignore  points  farther  than  this
DECIM_TARGET = 150000    #  thin  slices  are  sparse  ->  keep  more  pts

D_LIST = [2, 4, 6, 10, 15, 20, 30, 40, 60, 80, 100, 120, 150, 180]
SLICE_M = 0.05             # default  slice  thickness  [  D -  SLICE_M,  D  ]


def bags():
    return sorted(glob.glob(os.path.join(DATASET, "*", "*.db3")))


def pc_data(blob):
    raw = np.frombuffer(blob, np.uint8)[200:].reshape(-1, 26)
    x = raw[:, 0:4].view(np.float32).reshape(-1)
    y = raw[:, 4:8].view(np.float32).reshape(-1)
    s = raw[:, 8:12].view(np.float32).reshape(-1)
    k = ~((x == 0) & (y == 0) & (s == 0)) & np.isfinite(x) & np.isfinite(y)
    return x[k], y[k], s[k]


def prep(x, y):
    """forward  points,  decimated,  with  u-projection  ready"""
    f = -x
    m = (f > 0.1) & (np.abs(y) < 2 * Y_HALF)
    x, y, f = x[m], y[m], f[m]
    r = np.hypot(x, y)
    m2 = r <= R_BEYOND_MAX
    x, y, f, r = x[m2], y[m2], f[m2], r[m2]
    step = max(1, len(r) // DECIM_TARGET)
    if step > 1:
        x, y, f, r = x[::step], y[::step], f[::step], r[::step]
    return f, y, r


def section_line(f, y, D, slice_m=SLICE_M):
    """orthogonal  projection  onto  the  plane  x  =  -D  (  the  vertical
    plane  perpendicular  to  the  tunnel  axis  )  of  the  slab  whose
    axial  coordinate  lies  in  [D  -  slice_m,  D]  :  u  =  y  ."""
    m = (f >= D - slice_m) & (f <= D)
    u = y[m]
    b = ((u + Y_HALF) / BIN_M + 1e-6).astype(np.int32)  #  epsilon:  bin  edges
    ok = (b >= 0) & (b < N_BINS)
    return np.bincount(b[ok], minlength=N_BINS).astype(np.int32)


def edges(line):
    """(left,  right)  edge  of  the  section  in  metres  (  None  if  empty)"""
    idx = np.nonzero(line)[0]
    if len(idx) < 5:
        return None
    l = -Y_HALF + idx.min() * BIN_M
    rr = -Y_HALF + idx.max() * BIN_M
    return l, rr


def draw_section(line, hpx=160, wpx=900):
    """1-D  profile  ->  filled  B/W  area  image  (  one  tunnel  section  )"""
    c = line.astype(np.float64)
    mx = c.max()
    if mx <= 0:
        img = np.zeros((hpx, wpx), np.uint8)
        return img
    #  log  scale  so  sparse  bumps  are  visible  next  to  the  dense  floor
    norm = np.log1p(c) / np.log1p(mx)
    img = np.zeros((hpx, wpx), np.uint8)
    img[:] = 0
    xs = np.linspace(0, wpx - 1, N_BINS)
    top = (hpx - 1 - norm * (hpx - 20)).astype(int)
    for xi, ti in zip(xs, top):
        xi = int(round(xi))
        if 0 <= xi < wpx:
            img[int(ti):, xi] = 255
    return img


def render_sheet(sections, D_list, title, save_path):
    """stack  all  D  sections  vertically  with  edge  marks  +  width  text"""
    row_h, row_w = 150, 900
    lab_w = 330
    canvas = np.zeros((row_h * len(D_list) + 60, row_w + lab_w + 40), np.uint8)
    canvas[:] = 0
    cv2.putText(canvas, title, (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
    for i, (D, (line, lr)) in enumerate(sections):
        y0 = 50 + i * row_h
        sec = draw_section(line, hpx=row_h - 10, wpx=row_w)
        canvas[y0:y0 + row_h - 10, lab_w:lab_w + row_w] = sec
        #  lateral  axis  ticks  (  every  2  m  )
        for j in range(0, N_BINS, int(2 / BIN_M)):
            xx = lab_w + int(j / N_BINS * row_w)
            cv2.line(canvas, (xx, y0 + row_h - 10), (xx, y0 + row_h - 6), 120, 1)
            cv2.putText(canvas, f"{-Y_HALF + j * BIN_M:.0f}",
                        (xx - 6, y0 + row_h + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, 120, 1)
        #  D  label  +  width
        if lr:
            l, r = lr
            txt = f"D = {D:3d} m    width  {r - l:5.2f} m    centre  {(r + l) / 2:+6.2f} m"
        else:
            txt = f"D = {D:3d} m    (  no  returns  in  the  slab  )"
        cv2.putText(canvas, txt, (20, y0 + row_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        #  edge  markers  on  the  section
        if lr:
            l, r = lr
            for uu in (l, r):
                xx = lab_w + int((uu + Y_HALF) / (2 * Y_HALF) * row_w)
                cv2.line(canvas, (xx, y0), (xx, y0 + row_h - 10), 200, 1)
    if save_path:
        cv2.imwrite(save_path, canvas)
    return canvas


def width_plot(widths, D_list, title, save_path):
    """median  tunnel  width  vs  D  for  all  frames"""
    W, H = 900, 500
    img = np.zeros((H, W), np.uint8)
    cv2.putText(img, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    #  axes
    x0, x1, y0, y1 = 70, W - 30, 40, H - 60
    cv2.rectangle(img, (x0, y0), (x1, y1), 150, 1)
    dmax = max(D_list)
    wmax = 30.0
    for dd in D_list:
        xx = x0 + int(dd / dmax * (x1 - x0))
        cv2.line(img, (xx, y1), (xx, y1 + 5), 150, 1)
        cv2.putText(img, f"{dd}", (xx - 8, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 150, 1)
    for wv in range(0, 31, 5):
        yy = y1 - int(wv / wmax * (y1 - y0))
        cv2.line(img, (x0 - 5, yy), (x0, yy), 150, 1)
        cv2.putText(img, f"{wv}m", (10, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 150, 1)
    #  median  width  per  D
    pts = []
    for dd, ws in zip(D_list, widths):
        ws = np.array([w for w in ws if w is not None], np.float64)
        if len(ws) > 10:
            pts.append((dd, float(np.median(ws))))
    for i in range(1, len(pts)):
        (d1, w1), (d2, w2) = pts[i - 1], pts[i]
        cv2.line(img, (x0 + int(d1 / dmax * (x1 - x0)),
                       y1 - int(w1 / wmax * (y1 - y0))),
                 (x0 + int(d2 / dmax * (x1 - x0)),
                  y1 - int(w2 / wmax * (y1 - y0))), 255, 2)
    for dd, wv in pts:
        xx = x0 + int(dd / dmax * (x1 - x0))
        yy = y1 - int(wv / wmax * (y1 - y0))
        cv2.circle(img, (xx, yy), 4, 255, -1)
    if save_path:
        cv2.imwrite(save_path, img)


def process_bag(path, D_list, nstack, step, n_sheets=6, slice_m=SLICE_M):
    name = os.path.basename(os.path.dirname(path))
    out_dir = os.path.join(DIR, "tunnel_section")
    os.makedirs(out_dir, exist_ok=True)
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    msgs = con.execute("SELECT data FROM messages ORDER BY timestamp").fetchall()
    con.close()
    n = len(msgs)

    def stack_at(i):
        acc = {d: np.zeros(N_BINS, np.int32) for d in D_list}
        for j in range(i, min(i + nstack, n)):
            x, y, s = pc_data(msgs[j][0])
            f, y_, r = prep(x, y)
            for d in D_list:
                acc[d] += section_line(f, y_, d, slice_m)
        return acc

    #  1)  contact  sheets  along  the  corridor  (  evenly  spaced  )
    n_sheets = max(1, min(n_sheets, n // max(1, step)))
    if n_sheets > 1:
        picks = [int(round(k * (n - 1) / (n_sheets - 1)))
                 for k in range(n_sheets)]
    else:
        picks = [0]
    for k, i in enumerate(picks):
        acc = stack_at(i)
        lines = [(d, (acc[d], edges(acc[d]))) for d in D_list]
        render_sheet(lines, D_list,
                     f"{name}   frames  {i}..{i + nstack - 1}  of  {n}   "
                     f"(  position  {k + 1}/{n_sheets}  along  corridor  )",
                     os.path.join(out_dir, f"sheet_{name}_s{k + 1}.png"))

    #  2)  tunnel  width  vs  D  over  the  whole  bag
    width_rows = []
    widths_per_D = {d: [] for d in D_list}
    i = 0
    while i < n:
        acc = stack_at(i)
        for d in D_list:
            lr = edges(acc[d])
            widths_per_D[d].append(None if lr is None else lr[1] - lr[0])
            if lr:
                width_rows.append((i, d, lr[0], lr[1], lr[1] - lr[0],
                                   (lr[1] + lr[0]) / 2))
        i += step

    width_plot([widths_per_D[d] for d in D_list], D_list,
               f"{name}   tunnel  width  vs  distance  (  median  over  "
               f"{len(widths_per_D[D_list[0]])}  samples  )",
               os.path.join(out_dir, f"width_{name}.png"))

    with open(os.path.join(out_dir, f"width_{name}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "D_m", "left_m", "right_m", "width_m",
                   "centre_m"])
        for row in width_rows:
            w.writerow([row[0], row[1], f"{row[2]:.3f}", f"{row[3]:.3f}",
                        f"{row[4]:.3f}", f"{row[5]:.3f}"])
    print(f"==  {name}:  {len(picks)}  sheets  +  {len(width_rows)}  width  "
          f"rows  ->  tunnel_section/")


#  ============================================================================
#   OCCLUSION  MODEL  (  uses  the  vertical  references  ):
#
#     sensor  at  z = +1.075  m   (  1.075  m  above  the  rail  head  )
#     object  "  on  the  rail  ":   base  z  =  -1.075,   top  z  =  -1.075 + h
#     object  "  from  ceiling  ":  base  z  =  Htop - h,  top  =  Htop  (  max  )
#
#   A  beam  from  the  sensor  to  a  floor  point  at  distance  D  has  height
#
#        z(  d  )  =  1.075  *  (  1  -  2*d/D  )
#
#   at  the  object  distance  d  =  Do.  The  object  (  lateral  band  y  in  )
#   intercepts  it  when  its  vertical  span  contains  z(  Do  ):
#
#     on-rail  object   :  for  h  >=  1.075  *  (  1  -  Do/D  )
#                          (  i.e.  any  on-rail  box  occludes  the  floor  )
#     ceiling  object   :  never  (  its  span  is  above  the  floor  beam  )
#
#   =>  in  the  2.5-D  section  an  on-rail  object  leaves  a  notch  in  the
#      corridor  band  behind  it  (  D  >  Do  )  because  the  floor  returns
#      die;  a  ceiling  object  leaves  the  floor  band  intact  but  removes
#      the  ceiling  population  (  the  band  only  weakens  partially  ).
#
#   We  therefore  measure  the  band  strength  vs  D  per  lateral  bin  and
#   flag  drops  >=  50%  sustained  over  >=  2  consecutive  D  slices  :
#   a  drop  behind  a  detected  blob  =  an  occluding  object  there  .
#  ============================================================================


def band_strength(msgs, nstack, D_dense, step,
                  y_min=-3.5, y_max=3.5, n_y=71):
    """occupancy  of  the  corridor  band  [  y_min,  y_max  ]  per
    (  D  slice  ,  y  bin  )  for  the  whole  bag,  normalised  per  y  bin"""
    n = len(msgs)
    acc = np.zeros((len(D_dense), n_y), np.float32)
    i0 = 0
    while i0 < n:
        for j in range(i0, min(i0 + nstack, n)):
            x, y, s = pc_data(msgs[j][0])
            f, y_, r = prep(x, y)
            for di, d in enumerate(D_dense):
                #  same  model  as  section_line  :  slab  on  f,  u  =  y
                m = ((f >= d - SLICE_M) & (f <= d)
                     & (y_ >= y_min) & (y_ < y_max))
                if not m.any():
                    continue
                b = ((y_[m] - y_min) / (y_max - y_min) * n_y
                     + 1e-9).astype(np.int32)
                ok = (b >= 0) & (b < n_y)
                acc[di] += np.bincount(b[ok], minlength=n_y)
        i0 += max(1, step)
    #  normalise  per  D  row  :  far  slices  are  globally  sparser  than
    #  near  ones,  and  that  must  not  look  like  an  occlusion
    row = np.median(acc, axis=1)
    row[row == 0] = 1
    return acc / row[:, None]


def find_notches(strength, D_dense, y_min, y_max,
                 dip=0.4, min_run=3, pre=0.8):
    """occlusion  onset  per  y  bin.

    strength[di, j]  is  normalised  by  its  D-row  median,  i.e.  it  is  the
    relative  band  strength  of  y  bin  j  inside  slice  di  (  ~1  =  normal
    floor  density  for  that  D  ).  An  on-rail  object  at  (  Do,  y  )  kills
    the  floor  behind  it:  the  profile  of  y  bin  j  is  a  step  function  —
    >=  pre  before  Do,  <=  dip  for  all  D  beyond  Do.  We  return
    [(  D_onset,  y  )]  for  every  bin  showing  such  a  step  ."""
    out = []
    ny = strength.shape[1]
    for j in range(ny):
        prof = strength[:, j]
        #  onset  =  first  D  where  the  dip  starts  AND  the  profile  was
        #  healthy  right  before  it
        onset = None
        d = 0
        for i in range(1, len(prof)):
            if prof[i] <= dip:
                d += 1
                if d == min_run and onset is None:
                    before = prof[max(0, i - min_run - 2):i]
                    if len(before) and np.median(before) >= pre:
                        onset = i
            else:
                d = 0
        if onset is not None:
            #  require  the  dip  to  persist  to  (  at  least  )  the  end  of
            #  the  range  :  occlusion  behind  a  box  never  heals
            tail = prof[onset + min_run:]
            if tail.size and np.median(tail) <= dip * 1.5:
                yv = y_min + (j + 0.5) / ny * (y_max - y_min)
                out.append((D_dense[onset], yv))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="*",
                    help="bag  names  (  default:  all  )")
    ap.add_argument("--D", type=int, nargs="+", default=D_LIST,
                    help="distances  [m]  for  the  sections")
    ap.add_argument("--nstack", type=int, default=3,
                    help="stack  N  consecutive  frames  per  section")
    ap.add_argument("--step", type=int, default=10,
                    help="width  sample  per  N  frames")
    ap.add_argument("--sheets", type=int, default=6,
                    help="number  of  contact  sheets  per  bag")
    ap.add_argument("--slice", type=float, default=SLICE_M,
                    help="slice  thickness  [m]:  section  =  [D-slice,  D]")
    ap.add_argument("--occlude", action="store_true",
                    help="band  strength  vs  D  +  occlusion  notches")
    ap.add_argument("--nstack2", type=int, default=3,
                    help="frames  per  strength  sample")
    args = ap.parse_args()

    files = [p for p in bags()
             if not args.bags or any(b in p for b in args.bags)]
    for path in files:
        process_bag(path, sorted(args.D), args.nstack, args.step,
                    args.sheets, args.slice)
        if args.occlude:
            name = path.rsplit("/", 1)[-1].replace(".db3", "")
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            msgs = con.execute(
                "SELECT data FROM messages ORDER BY timestamp").fetchall()
            con.close()
            D_dense = list(range(2, 61, 2))
            st = band_strength(msgs, args.nstack2, D_dense, args.step)
            nt = find_notches(st, D_dense, -3.5, 3.5)
            nt2 = []
            for d, yv in nt:
                if not nt2 or abs(d - nt2[-1][0]) > 1 or \
                        abs(yv - nt2[-1][1]) > 0.5:
                    nt2.append((d, yv))
            odir = os.path.join(DIR, "tunnel_section")
            os.makedirs(odir, exist_ok=True)
            outc = os.path.join(odir, f"occlude_{name}.csv")
            with open(outc, "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["D_start_m", "y_m"])
                for d, yv in nt2:
                    wr.writerow([d, round(yv, 2)])
            print(f"==  {name}:  occlusion  sweep  ->  {outc}  "
                  f"  (  {len(nt2)}  notch  zones  )")


if __name__ == "__main__":
    main()
