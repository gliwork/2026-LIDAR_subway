#!/usr/bin/env python3
"""
Synthetic  obstacle  map  from  the  LiDAR  bags  (  B/W  ).

All  the  scenario  obstacles  in  the  dataset  are  SYNTHETIC  boxes
(  2x2  centre,  0.3x0.3  on  rail  /  at  gauge  edge  /  outside  gauge,
2x2  at  edge  /  outside  /  above  the  profile,  2x0.2  on  rail,
0.05-m-wide  hanging  from  the  ceiling  ).  A  box  in  the  tunnel  frame
is  a  rectangular  region  of  space  ->  in  the  2.5-D  data  it  shows  up
as  a  filled  rectangle  in  the  (  D,  y  )  map  :

    D  (  range  bin  ,  2  m  )  x  y  (  lateral  bin  ,  0.1  m  )

Outputs  (  per  bag,  in  tunnel_section/  ):
  omap_<bag>.png        the  (  D,  y  )  occupancy  map  of  the  whole
                        bag  (  log  scale,  B/W  )  with  detected  boxes
  oboxes_<bag>.csv      detected  rectangles:  D,  y,  size  class

usage:
  python3 obstacle_map.py                            #  all  bags
  python3 obstacle_map.py doubleT_obstacle           #  one  bag
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

D_STEP = 2.0            # range  bin  [m]
R_MAX = 180.0           # map  range  [m]
Y_HALF = 3.5            # lateral  half  width  [m]
Y_STEP = 0.1            # lateral  bin  [m]
N_D = int(R_MAX / D_STEP)           # 90
N_Y = int(2 * Y_HALF / Y_STEP)      # 70


def pois_sig(lam, alpha=0.01):
    """min  count  k  such  that  P(  Poisson(  lam  )  >=  k  )  <=  alpha
    (  the  window  must  hold  a  real  excess  over  the  expected
    random  background  fall-in  ,  not  just  a  ratio  spike  )"""
    if lam <= 0:
        return 1
    k = int(lam)
    while k < 200:
        #  upper  tail  P(  X  >=  k  )
        p = 0.0
        kk = k
        while kk < k + 120:
            import math
            p += math.exp(-lam + kk * math.log(lam)
                          - math.lgamma(kk + 1))
            if p > 1:
                p = 1
            kk += 1
        if p <= alpha:
            return k
        k += 1
    return 200

#  expected  box  footprints  :  (  L  along  D,  W  lateral  )  [m]
#  box  classes  (  L  along  D  ,  W  lateral  )  [m]  :
BOXES = [
    ("2x2", 2.0, 2.0),
    ("2x0.2", 2.0, 0.2),
    ("0.3x0.3", 0.3, 0.3),
]
#  the  0.05-m  hanging  box  is  sub-bin  wide  ->  reported  as  isolated
#  hot  pixels  instead  of  a  window  search


def pc_data(blob):
    p = np.frombuffer(blob, np.uint8)[200:].reshape(-1, 26)
    x = p[:, 0:4].view(np.float32).reshape(-1)
    y = p[:, 4:8].view(np.float32).reshape(-1)
    k = ~((x == 0) & (y == 0)) & np.isfinite(x) & np.isfinite(y)
    return x[k], y[k]


def build_map(msgs, nstack=1, step=1):
    """(  N_D,  N_Y  )  count  grid  :  range  bin  x  lateral  bin"""
    grid = np.zeros((N_D, N_Y), np.int32)
    n = len(msgs)
    i = 0
    while i < n:
        for j in range(i, min(i + nstack, n)):
            x, y = pc_data(msgs[j][0])
            f = -x
            m = (f > 0.3) & (f < R_MAX) & (y >= -Y_HALF) & (y < Y_HALF)
            if not m.any():
                continue
            db = (f[m] / D_STEP).astype(np.int32)
            yb = ((y[m] + Y_HALF) / Y_STEP).astype(np.int32)
            ok = (db < N_D) & (yb >= 0) & (yb < N_Y)
            idx = db[ok] * N_Y + yb[ok]
            grid += np.bincount(idx, minlength=N_D * N_Y) \
                .reshape(N_D, N_Y)
        i += max(1, step)
    return grid


def find_boxes(grid, fill=0.6, e_th=2.0):
    """sliding  rectangle  search  for  the  expected  box  sizes.

    A  synthetic  box  is  a  compact  local  EXCESS  of  returns  :
      *  local  background  =  per-y  sliding  median  over  +  -  10  D
        bins  (  a  global  median  fails  :  the  near  field  is  10  x
        denser  than  the  far  field  )
      *  a  box  stands  out  from  its  D  neighbours  (  a  long  structure
        like  a  platform  edge  has  saturated  neighbours  ->  rejected  )
      *  the  window  (  L  x  W  )  must  be  mostly  hot  (  E  >=  e_th  )"""
    found = []
    #  local  background  :  sliding  median  in  D  per  y  bin
    bg = np.zeros_like(grid, np.float32)
    W2 = 10
    for j in range(N_Y):
        col = grid[:, j].astype(np.float32)
        for i in range(N_D):
            a, b = max(0, i - W2), min(N_D, i + W2 + 1)
            bg[i, j] = np.median(col[a:b])
    bg = np.maximum(bg, 1.0)
    E = grid / bg

    for name, L, W in BOXES:
        wD = max(1, int(round(L / D_STEP)))
        wY = max(1, int(round(W / Y_STEP)))
        hot = (E >= e_th).astype(np.int32)
        Sg = np.zeros((N_D + 1, N_Y + 1), np.int64)
        Sg[1:, 1:] = grid.astype(np.int64).cumsum(0).cumsum(1)
        Sh = np.zeros((N_D + 1, N_Y + 1), np.int64)
        Sh[1:, 1:] = hot.cumsum(0).cumsum(1)

        def wsum(S, i0, j0):
            i1, j1 = min(N_D, i0 + wD), min(N_Y, j0 + wY)
            return S[i1, j1] - S[i0, j1] - S[i1, j0] + S[i0, j0]

        rowsum = grid.sum(axis=1)
        hits = []
        for i0 in range(0, N_D - wD + 1):
            i1 = i0 + wD
            #  expected  random  fall-in  of  the  window  from  its  rows
            lam = float(rowsum[i0:i1]) * (wY / N_Y)
            alpha = 0.001 if name != "2x2" else 0.005
            need = max(2, wD * wY // 3, pois_sig(lam, alpha))
            for j0 in range(0, N_Y - wY + 1):
                sh = wsum(Sh, i0, j0)
                if sh / float(wD * wY) < fill:
                    continue
                sg = wsum(Sg, i0, j0)
                if sg < need:
                    continue
                #  isolation  in  D  :  the  rows  next  to  the  window
                #  (  same  y  span  )  must  NOT  be  saturated
                iso = True
                for (ia, ib) in ((i0 - 2, i0), (i0 + wD, i0 + wD + 2)):
                    ia, ib = max(0, ia), min(N_D, ib)
                    if ia >= ib:
                        continue
                    sn = wsum(Sh, ia, j0) / float((ib - ia) * wY)
                    if sn >= fill:
                        iso = False
                        break
                if not iso:
                    continue
                Dc = (i0 + wD / 2) * D_STEP
                yc = -Y_HALF + (j0 + wY / 2) * Y_STEP
                hits.append((Dc, yc, sh))
        hits.sort(key=lambda h: -h[2])
        kept = []
        for Dc, yc, s in hits:
            #  distance  limits  :  beyond  these  ranges  the  class  is
            #  just  Poisson  noise  of  the  clumpy  wall  background
            if Dc < 15:                     # vehicle  body  zone
                continue
            if name == "2x2" and (Dc > 150 or s < 20):
                continue
            if name in ("2x0.2", "0.3x0.3") and (Dc > 90 or s < 6):
                continue
            if all(abs(Dc - kd) > D_STEP or abs(yc - ky) > 0.4
                   for kd, ky, _ in kept):
                kept.append((Dc, yc, s))
        for Dc, yc, s in kept:
            found.append((name, Dc, yc))

    #  sub-bin  0.05-m  class  :  isolated  very  hot  pixels  (  near  /
    #  mid  range  only  :  out  far  a  few  points  is  just  noise  )
    for i in range(int(15 / D_STEP), int(60 / D_STEP)):
        need = max(3, pois_sig(float(rowsum[i]) * (1 / N_Y), 0.001))
        for j in range(N_Y):
            if E[i, j] < 4.0 or grid[i, j] < need:
                continue
            #  D  isolation  (  the  box  is  short  )
            nb = [E[k, j] for k in (i - 1, i + 1) if 0 <= k < N_D]
            if nb and np.median(nb) >= 3.0:
                continue
            found.append(("0.05", (i + 0.5) * D_STEP,
                          -Y_HALF + (j + 0.5) * Y_STEP))
    return found
    #  drop  a  small  box  if  it  sits  inside  a  large  one  hit  nearby
    out = []
    for name, Dc, yc in found:
        if name in ("0.3x0.3", "0.05", "2x0.2"):
            if any(big in ("2x2",) and abs(Dc - bd) < 2 and abs(yc - by) < 2
                   for big, bd, by in found):
                continue
        out.append((name, Dc, yc))
    return out


def find_structures(grid, min_len=8.0, e_th=2.0):
    """long  lateral  bands  (  >  min_len  m  in  D  )  :  these  are
    corridor  fixtures  (  platform  edges  ,  wall  offsets  ,  gates  )
    ,  NOT  the  short  synthetic  boxes"""
    bg = np.zeros_like(grid, np.float32)
    for j in range(N_Y):
        col = grid[:, j].astype(np.float32)
        for i in range(N_D):
            a, b = max(0, i - 10), min(N_D, i + 10 + 1)
            bg[i, j] = np.median(col[a:b])
    bg = np.maximum(bg, 1.0)
    E = grid / bg
    out = []
    rowsum = grid.sum(axis=1)
    for j in range(N_Y):
        yc = -Y_HALF + (j + 0.5) * Y_STEP
        run = 0
        i0 = 0
        for i in range(0, N_D + 1):
            if i < N_D:
                need = max(3, pois_sig(float(rowsum[i]) * (3 / N_Y),
                                       0.001))
                hot = E[i, j] >= e_th and grid[i, j] >= need
            else:
                hot = False
            if hot and run == 0:
                i0 = i
                run = 1
            elif hot:
                run += 1
            else:
                if run * D_STEP >= min_len:
                    out.append((yc, i0 * D_STEP, (i - 1) * D_STEP))
                run = 0
    return out


def render(grid, found, title, save_path):
    g = np.log1p(grid.astype(np.float32))
    mx = g.max()
    if mx > 0:
        g = g / mx * 255
    img = cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    H, W = img.shape[:2]

    #  labels
    cv2.putText(img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    for d in range(0, N_D, 5):
        ypx = int(d / N_D * H)
        cv2.line(img, (0, ypx), (14, ypx), (90, 90, 90), 1)
        cv2.putText(img, str(int(d * D_STEP)), (2, ypx + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
                    cv2.LINE_AA)
    for j in range(0, N_Y, 10):
        xpx = int(j / N_Y * W)
        lab = int(-Y_HALF + j * Y_STEP)
        cv2.putText(img, str(lab), (xpx - 6, H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1,
                    cv2.LINE_AA)

    #  box  marks
    for name, Dc, yc in found:
        col = (0, 255, 255)
        boxdict = {b[0]: (b[1], b[2]) for b in BOXES}
        if name in boxdict:
            L, Wm = boxdict[name][0], boxdict[name][1]
        else:                      # 0.05  hot  pixel  class
            L, Wm = 0.5, 0.1
        x0 = int((yc - Wm / 2 + Y_HALF) / (2 * Y_HALF) * W)
        x1 = int((yc + Wm / 2 + Y_HALF) / (2 * Y_HALF) * W)
        y0 = int((Dc - L / 2) / R_MAX * H)
        y1 = int((Dc + L / 2) / R_MAX * H)
        cv2.rectangle(img, (max(0, x0), max(0, y0)),
                      (min(W, x1), min(H, y1)), col, 1)
        cv2.putText(img, name, (max(0, x0), max(0, y0) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    cv2.imwrite(save_path, img)


def process_bag(path, nstack, step, fill, e_th):
    name = path.rsplit("/", 1)[-1].replace(".db3", "")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    msgs = con.execute(
        "SELECT data FROM messages ORDER BY timestamp").fetchall()
    con.close()
    grid = build_map(msgs, nstack, step)
    found = find_boxes(grid, fill=fill, e_th=e_th)
    structs = find_structures(grid)
    odir = os.path.join(DIR, "tunnel_section")
    os.makedirs(odir, exist_ok=True)
    render(grid, found, f"{name}   (  D  down,  y  right  )",
           os.path.join(odir, f"omap_{name}.png"))
    outc = os.path.join(odir, f"oboxes_{name}.csv")
    with open(outc, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["box", "D_m", "y_m"])
        for name_, Dc, yc in sorted(found, key=lambda t: t[1]):
            wr.writerow([name_, round(Dc, 1), round(yc, 2)])
        for yc, d0, d1 in structs:
            wr.writerow(["structure", f"{d0:.0f}-{d1:.0f}", round(yc, 2)])
    print(f"==  {name}:  {len(found)}  box  hits,  "
          f"{len(structs)}  long  structures")
    for name_, Dc, yc in sorted(found, key=lambda t: t[1]):
        print(f"      box  {name_:>8s}   D = {Dc:6.1f} m   y = {yc:+5.2f} m")
    for yc, d0, d1 in sorted(structs):
        print(f"      band  y = {yc:+5.2f} m   D = {d0:5.0f} - {d1:5.0f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="*")
    ap.add_argument("--nstack", type=int, default=1)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--fill", type=float, default=0.6,
                    help="min  fraction  of  hot  cells  in  the  window")
    ap.add_argument("--eth", type=float, default=2.0,
                    help="excess  threshold  over  the  per-y  background")
    args = ap.parse_args()
    files = [p for p in sorted(glob.glob(os.path.join(DATASET, "*", "*.db3")))
             if not args.bags or any(b in p for b in args.bags)]
    for path in files:
        process_bag(path, args.nstack, args.step, args.fill, args.eth)


if __name__ == "__main__":
    main()
