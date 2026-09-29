#!/usr/bin/env python3
"""
shadow_detect.py  v2  —  extract  the  synthetic  boxes  from  the
(  y  ,  z  )  "  shadow  "  screen  series  built  by  lidar_slices.py  .

Model
-----
Work  in  ANGULAR  coordinates  as  seen  from  the  sensor  :

    a  =  y  /  f  ,   b  =  z  /  f     (  f  =  -x  forward  distance  ;
                                         z  pre-scaled  by  --z-scale  )

A  point  projected  on  the  screen  at  distance  D  is  (  u  ,  v  )
=  (  D  *  a  ,  D  *  b  )  .  A  box  at  distance  d  has  therefore  a
FIXED  angular  footprint  that  does  not  move  with  D  :  its  image
on  the  screens  just  grows  linearly  ,  (  u  ,  v  )  =  D  *
(  a  ,  b  )  .

For  every  screen  we  bin  the  points  with  f  >  D  +  gap  into
(  a  ,  b  )  ,  giving  a  per-cell  count  M  (  D  )  .

KEY  OBSERVATION
----------------
A  point  keeps  the  same  (  a  ,  b  )  on  all  screens  ,  so  a
cell  stays  filled  for  as  long  as  the  points  inside  it  are
beyond  the  screen  .  Hence  :

  *  a  BOX  (  a  compact  f  -  range  )  makes  its  cells  fall  in
    a  SHARP  STEP  :  M  is  flat  ,  then  drops  to  ~0  in  one  or
    two  screens  when  D  +  gap  passes  its  far  side  d2  ;

  *  the  corridor  FLOOR  and  WALLS  have  a  wide  f  -  range  :
    their  cells  decay  as  a  SMOOTH  RAMP  (  a  few  percent  per
    screen  )  —  the  floor  band  literally  slides  towards  the
    vanishing  point  in  (  a  ,  b  )  space  ;

  *  a  SPARSE  region  (  far  wall  ,  few  returns  )  has  no
    structure  at  all  .

The  detector  :
  1.  marks  cells  that  were  bright  at  some  screen  and  are  empty
      in  the  last  screens  ;
  2.  for  each  such  cell  finds  the  SHARP  DROP  screen  k  (  the
      first  screen  where  M  falls  to  <=  25  %  of  the  previous
      one  ,  preceded  by  at  most  2  partial  drops  —  this  is
      what  rejects  the  ramps  )  ;
  3.  groups  the  drop  cells  :  same  drop  screen  +  spatially
      contiguous  =  one  box  (  all  the  faces  of  a  box  drop  in
      the  same  one  -  two  screens  ,  while  two  boxes  at  the
      same  distance  but  different  angles  are  separate  clusters  )  ;
  4.  the  points  inside  the  box  region  whose  f  falls  in  the
      drop  window  are  the  BOX  FACES  (  a  background  point  of
      that  direction  is  either  in  front  of  the  box  and  drops
      out  much  earlier  ,  or  behind  it  and  does  not  exist  )  :

        d1  ,  d2  =  min  /  max  f  of  those  points
        width     =  y  -  range  ,  height  =  z  -  range  ,
        depth     =  d2  -  d1

  5.  cross-checks  :  the  drop  position  must  equal  d2  ;  the
      linear  growth  fit  d  =  width  /  (  angular  width  )  must
      agree  with  d2  ;  size  must  be  plausible  .

Outputs
-------
    <outdir>/shadow_<bag>.csv
    <outdir>/shadow_<bag>_D<xx>.png      annotated  screens
    console  tables  +  cross-bag  grouping  (  the  same  box  in
    several  bags  =  a  common  scenario  fixture  )

Usage
-----
    python3  shadow_detect.py  FILE  [  FILE  ...  ]
        --step  5  --gap  5  --dmax  120
        --z-scale  0.01          (  for  the  hackathon  0..255  slot  )
        --min-d  15  --max-size  8  --min-pts  30
        --outdir  shadows
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_slices as ls

ANG_BIN = 0.001            #  rad  (  ~  0.057  deg  )

F = Y = Z = None           #  filled  by  process  (  used  by  _assess  )


# ------------------------------------------------------------ angular
def build_ang_screens(f, y, z, thresholds, A, Bmin, Bmax):
    """H  [  nD  ,  NB  ,  NA  ]  :  per  screen  ,  the  points  with
    f  >  threshold  binned  by  (  a  =  y  /  f  ,  b  =  z  /  f  )  ."""
    NA = int(round(2 * A / ANG_BIN))
    NB = int(round((Bmax - Bmin) / ANG_BIN))
    H = np.zeros((len(thresholds), NB, NA), np.int32)
    for k, thr in enumerate(thresholds):
        m = f > thr
        if not m.any():
            continue
        af = y[m] / f[m]
        bf = z[m] / f[m]
        inb = (af >= -A) & (af < A) & (bf >= Bmin) & (bf < Bmax)
        if not inb.any():
            continue
        ia = ((af[inb] + A) / (2 * A) * NA + 1e-6).astype(np.int32)
        ib = ((bf[inb] - Bmin) / (Bmax - Bmin) * NB + 1e-6).astype(np.int32)
        cnt = np.bincount(ia + ib * NA, minlength=NB * NA)
        H[k] = cnt[:NB * NA].reshape(NB, NA)
    return H, NA, NB


def cell_drops(H, cand, n_screens, ramp_win=6, ramp_ratio=1.5):
    """for  every  candidate  cell  :  the  screen  k  of  its  sharp
    final  drop  ,  or  -1  .

    A  sharp  drop  at  k  :
        M  [  k  ]  <=  max  (  2  ,  0.25  *  M  [  k  -  1  ]  )
        M  [  k  -  1  ]  >=  5
    and  NOT  a  ramp  :  the  count  ramp_win  screens  earlier  must
    not  be  ramp_ratio  x  larger  than  M  [  k  -  1  ]  .  A  box
    (  narrow  f  -  range  )  stays  flat  before  the  drop  ;  the
    floor  /  wall  tracks  (  wide  f  -  range  )  decay  steadily  ,
    so  M  [  k  -  ramp_win  ]  >>  M  [  k  -  1  ]  .

    M  is  a  3x3  -  smoothed  per-cell  count  (  stabilises  sparse
    data  )  ."""
    M = ndimage.uniform_filter(H.astype(np.float64),
                               size=(1, 3, 3)) * 9.0
    kdrop = np.full((H.shape[1], H.shape[2]), -1, np.int16)
    for k in range(1, n_screens):
        prev = M[k - 1]
        is_drop = (cand & (prev >= 5) &
                   (M[k] <= np.maximum(2.0, 0.25 * prev)))
        ref = M[max(0, k - ramp_win)]
        is_ramp = ref > ramp_ratio * prev + 2.0
        new = is_drop & ~is_ramp & (kdrop == -1)
        kdrop[new] = k
    return kdrop


# ----------------------------------------------------------- detection
def _assess(S, a1, a2, b1, b2, thresholds, k_drop, a):
    """from  the  points  of  one  box  cluster  :  the  measured  box  ."""
    f, y, z = F, Y, Z
    #  the  drop  window  :  between  the  two  screens  that  bracket
    #  the  drop
    lo = thresholds[k_drop - 1] - 0.5 if k_drop > 0 else thresholds[0] - 1
    hi = thresholds[k_drop] + 0.5
    mS = S[(f[S] >= lo) & (f[S] <= hi)]
    if len(mS) < max(10, a.min_pts // 3):
        return None
    fs = f[mS]
    d1 = float(np.percentile(fs, 1))
    d2 = float(np.percentile(fs, 99))
    if d1 < a.min_d:
        return None
    if d2 - d1 > (hi - lo) + 0.5:
        return None
    if d2 > hi + 0.5 or d2 < lo - 0.5:
        return None

    w = float(np.percentile(y[mS], 99.5) - np.percentile(y[mS], 0.5))
    h = float(np.percentile(z[mS], 99.5) - np.percentile(z[mS], 0.5))
    if not (0.1 <= w <= a.max_size and 0.02 <= h <= a.max_size):
        return None

    #  linear  growth  fit  :  the  footprint  on  screen  is
    #  (  a2  -  a1  )  *  D  ->  d  =  width  /  (  a2  -  a1  )
    d2_growth = float(w / (a2 - a1)) if (a2 - a1) > 1e-6 else d2
    growth_ok = abs(d2_growth - d2) / max(1.0, d2) <= 0.25

    return dict(
        d1=d1, d2=d2, depth=d2 - d1,
        y1=float(np.percentile(y[mS], 0.5)),
        y2=float(np.percentile(y[mS], 99.5)), width=w,
        z1=float(np.percentile(z[mS], 0.5)),
        z2=float(np.percentile(z[mS], 99.5)), height=h,
        yc=float(np.median(y[mS])), zc=float(np.median(z[mS])),
        n=int(len(mS)),
        d2t_lo=lo, d2t_hi=hi, k_drop=k_drop,
        d2_growth=d2_growth, growth_ok=growth_ok,
        a1=a1, a2=a2, b1=b1, b2=b2, S=mS)


def _group_boxes(kdrop, H, thresholds, a, Bmin, Bmax):
    boxes = []
    ncells = float(H.shape[1] * H.shape[2])
    ks = sorted(set(kdrop[kdrop >= 0].tolist()))
    for k in ks:
        mask = (kdrop == k)
        if mask.sum() < a.min_cells:
            continue
        if mask.sum() > 0.05 * ncells:
            continue
            #  a  global  event  (  the  tunnel  ends  inside  dmax  )
            #  ,  not  a  box
        #  bridge  small  gaps  (  swiss  -  cheese  cells  of  a  face
        #  )  and  split  spatially  separated  boxes
        dil = ndimage.binary_dilation(mask, iterations=a.merge_w)
        lab, ncl = ndimage.label(dil, structure=np.ones((3, 3), np.int32))
        for ci in range(1, ncl + 1):
            cells = np.where((lab == ci) & mask)
            n_cells = len(cells[0])
            if n_cells < max(4, a.min_cells // 4):
                continue
            NA = H.shape[2]
            if n_cells > 0.15 * H.shape[1] * NA:
                continue
            a1 = cells[1].min() * ANG_BIN - a.ang_half
            a2 = (cells[1].max() + 1) * ANG_BIN - a.ang_half
            b1 = cells[0].min() * ANG_BIN + Bmin
            b2 = (cells[0].max() + 1) * ANG_BIN + Bmin
            box = _cluster_points(a1, a2, b1, b2, thresholds, k, a)
            if box is not None:
                boxes.append(box)
    #  merge  groups  whose  angular  ranges  touch  AND  whose  f
    #  -  ranges  are  close  (  the  front  and  the  back  face  of
    #  the  same  box  drop  in  adjacent  screens  )
    touch = 2 * ANG_BIN
    merged = []
    for b in boxes:
        for m in merged:
            close_f = (abs(b['d2'] - m['d1']) <= 1.5 * a.step and
                       abs(m['d2'] - b['d1']) <= 1.5 * a.step)
            overlap = (b['a1'] <= m['a2'] + touch and
                       m['a1'] <= b['a2'] + touch and
                       b['b1'] <= m['b2'] + touch and
                       m['b1'] <= b['b2'] + touch)
            if close_f and overlap:
                break
        else:
            merged.append(b)
            continue
        mS = np.concatenate([m['S'], b['S']])
        m.update({
            'S': mS,
            'd1': min(m['d1'], b['d1']), 'd2': max(m['d2'], b['d2']),
            'y1': min(m['y1'], b['y1']), 'y2': max(m['y2'], b['y2']),
            'z1': min(m['z1'], b['z1']), 'z2': max(m['z2'], b['z2']),
            'n': m['n'] + b['n'],
            'a1': min(m['a1'], b['a1']), 'a2': max(m['a2'], b['a2']),
            'b1': min(m['b1'], b['b1']), 'b2': max(m['b2'], b['b2']),
            'yc': float(np.median(Y[mS])), 'zc': float(np.median(Z[mS])),
            'width': float(np.percentile(Y[mS], 99.5) -
                           np.percentile(Y[mS], 0.5)),
            'height': float(np.percentile(Z[mS], 99.5) -
                           np.percentile(Z[mS], 0.5)),
            'd2t_lo': min(m['d2t_lo'], b['d2t_lo']),
            'd2t_hi': max(m['d2t_hi'], b['d2t_hi']),
        })
        m['depth'] = m['d2'] - m['d1']
        m['d2_growth'] = (m['width'] / (m['a2'] - m['a1'])
                          if (m['a2'] - m['a1']) > 1e-6 else m['d2'])
        m['growth_ok'] = abs(m['d2_growth'] - m['d2']) / max(1.0, m['d2']) \
            <= 0.25
    return merged


def _cluster_points(a1, a2, b1, b2, thresholds, k, a):
    """all  measured  points  inside  the  cluster  range  ->  box  ."""
    f, y, z = F, Y, Z
    af = y / f
    bf = z / f
    mS = ((af >= a1) & (af <= a2) & (bf >= b1) & (bf <= b2) &
          (f > 0) & (f <= thresholds[-1]))
    S = np.where(mS)[0]
    if len(S) < a.min_pts:
        return None
    #  split  at  big  f  -  gaps  (  one  box  behind  another  )
    Sf = np.sort(S)
    gaps = np.diff(f[Sf])
    cut = list(np.where(gaps > 2.0)[0] + 1) + [len(Sf)]
    prev = 0
    best = None
    for c in cut:
        part = Sf[prev:c]
        prev = c
        if len(part) < max(10, a.min_pts // 3):
            continue
        b = _assess(part, a1, a2, b1, b2, thresholds, k, a)
        if b is not None and (best is None or b['n'] > best['n']):
            best = b
    return best


def detect(f, y, z, thresholds, a, Bmin, Bmax):
    """run  the  whole  detector  ;  returns  (  boxes  ,  H  )  ."""
    H, NA, NB = build_ang_screens(f, y, z, thresholds,
                                  a.ang_half, Bmin, Bmax)
    peak = H.max(axis=0)
    late = H[-a.late_n:].sum(axis=0)
    cand = (peak >= a.t_peak) & (late <= np.maximum(
        2, (a.t_frac * peak).astype(int)))
    print(f"    angular  screens  :  {len(thresholds)}  x  {NB}  x  "
          f"{NA}  ,  candidate  cells  :  {int(cand.sum())}  (  "
          f"{100 * cand.mean():.1f}  %  )")
    kdrop = cell_drops(H, cand, len(H), a.ramp_win, a.ramp_ratio)
    print(f"    cells  with  a  sharp  final  drop  :  "
          f"{int((kdrop >= 0).sum())}  ")
    boxes = _group_boxes(kdrop, H, thresholds, a, Bmin, Bmax)
    boxes.sort(key=lambda b: -b['n'])
    return boxes, H


# ------------------------------------------------------------- render
def render_annot(G, D, boxes, path, yspan, zmin, zmax, z_kind, scale=4):
    g = np.log1p(G.astype(np.float32))
    img = np.zeros(G.shape, np.uint8)
    if g.max() > 0:
        img = (255 * g / g.max()).astype(np.uint8)
    img = img[::-1]
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                     interpolation=cv2.INTER_NEAREST)
    Hh, Ww = img.shape

    def px(u, v):
        x = int((u + yspan) / (2 * yspan) * Ww)
        yv = int((1 - (v - zmin) / (zmax - zmin)) * Hh)
        return max(0, min(Ww - 1, x)), max(0, min(Hh - 1, yv))

    title = f"screen  D  =  {D:5.1f}  m  (  box  shadows  /  blobs  )"
    cv2.putText(img, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                255, 1, cv2.LINE_AA)
    for k, b in enumerate(boxes):
        pts = b["S"]
        fsb = b["fsub"]
        keep = fsb > D
        if keep.any():
            u = Y[pts[keep]] * D / fsb[keep]
            v = Z[pts[keep]] * D / fsb[keep]
            u1, u2 = float(u.min()), float(u.max())
            v1, v2 = float(v.min()), float(v.max())
            thick = 2
        else:
            u1, u2 = b["a1"] * D, b["a2"] * D
            v1, v2 = b["b1"] * D, b["b2"] * D
            thick = 1
        p1 = px(u1, v2)
        p2 = px(u2, v1)
        cv2.rectangle(img, p1, p2, 255, thick)
        tag = f"box  {k + 1}  :  d  =  {b['d2']:.0f}  m  ,  " \
              f"{b['width']:.1f}  x  {b['height']:.1f}  m"
        if z_kind != "metric":
            tag += "  (  z  =  attr  )  "
        cv2.putText(img, tag, (p1[0] + 4, max(12, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


# --------------------------------------------------------------- main
def process(path, a, outdir):
    global F, Y, Z
    base = os.path.splitext(os.path.basename(path))[0]
    x, y, z = ls.load_any(path, a.max_frames, a.per_frame)
    z = z * a.z_scale
    z_kind = "metric" if a.z_scale == 1.0 else "attribute"
    f = x if a.forward_x else -x
    F, Y, Z = f, y, z

    fp = f[f > 0]
    dmax = a.dmax or min(200.0, float(np.percentile(fp, 99)))
    step = a.step
    gap = a.gap if a.gap is not None else step
    start = a.start or step

    yspan = a.yspan or min(8.0, max(3.0, float(np.percentile(
        np.abs(y), 99))))
    zmin = a.zmin if a.zmin is not None else float(
        np.percentile(z, 0.1)) - 0.2
    zmax = a.zmax if a.zmax is not None else float(
        np.percentile(z, 99.9)) + 0.2

    D_list, grids, npts = ls.build_slices(
        f, y, z, step, gap, start, dmax, yspan, zmin, zmax, a.bin_m)
    thresholds = [D + gap for D in D_list]
    print(f"  [{base}]  {len(f)}  pts  ,  screens  D  =  "
          f"{D_list[0]:.0f}  ..  {D_list[-1]:.0f}  m  "
          f"(  threshold  D  +  {gap:g}  )  ,  z  =  "
          f"[{zmin:.2f}  ,  {zmax:.2f}]  {z_kind}")
    boxes, H = detect(f, y, z, thresholds, a, a.bmin, a.bmax)
    for b in boxes:
        b["fsub"] = f[b["S"]]

    rows = []
    for k, b in enumerate(boxes):
        rows.append({
            "bag": base, "box_id": k + 1,
            "d1_m": round(b["d1"], 2), "d2_m": round(b["d2"], 2),
            "depth_m": round(b["depth"], 2),
            "y_min_m": round(b["y1"], 2), "y_max_m": round(b["y2"], 2),
            "width_m": round(b["width"], 2),
            "z_min": round(b["z1"], 3), "z_max": round(b["z2"], 3),
            "height": round(b["height"], 3),
            "y_center_m": round(b["yc"], 2),
            "z_center": round(b["zc"], 3),
            "n_pts": b["n"],
            "d2_transition": f"{b['d2t_lo']:.0f}..{b['d2t_hi']:.0f}",
            "d2_growth_fit_m": round(b["d2_growth"], 1),
            "growth_fit_ok": int(b["growth_ok"]),
            "z_kind": z_kind})
        print(f"    box  {k + 1}  :  d  =  {b['d1']:.1f}  ..  "
              f"{b['d2']:.1f}  m  (  depth  {b['depth']:.1f}  )  ,  "
              f"y  =  {b['y1']:.2f}  ..  {b['y2']:.2f}  m  "
              f"(  width  {b['width']:.2f}  )  ,  z  =  "
              f"{b['z1']:.2f}  ..  {b['z2']:.2f}  (  {b['height']:.2f}  "
              f"{z_kind}  )  ,  {b['n']}  pts  ,  transition  "
              f"{b['d2t_lo']:.0f}..{b['d2t_hi']:.0f}  ,  growth  fit  "
              f"{'OK' if b['growth_ok'] else 'n/a'}")

    os.makedirs(outdir, exist_ok=True)
    if rows:
        cp = os.path.join(outdir, f"shadow_{base}.csv")
        with open(cp, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"    csv  ->  {os.path.basename(cp)}")

    if not a.no_annotate:
        picked = set()
        for b in boxes:
            for D in D_list:
                if D < b["d1"] - 25:
                    picked.add(D)
                    break
            for D in D_list:
                if b["d1"] - 15 <= D < b["d1"] - 5:
                    picked.add(D)
            for D in D_list:
                if b["d2"] + gap + 5 <= D <= b["d2"] + gap + 30:
                    picked.add(D)
            for D in D_list:
                if D > b["d2"] + gap + 30:
                    picked.add(D)
                    break
        for D in sorted(picked)[:8]:
            p = os.path.join(outdir, f"shadow_{base}_D{D:05.1f}.png")
            render_annot(grids[D], D, boxes, p, yspan, zmin, zmax,
                         z_kind)
            print(f"    annotated  D  =  {D:5.1f}  m  ->  "
                  f"{os.path.basename(p)}")
    return rows


def cross_bag(all_rows):
    if not all_rows:
        print("\nno  boxes  detected  in  any  bag")
        return
    bags = sorted(set(r["bag"] for r in all_rows))
    print(f"\n  cross-bag  grouping  (  same  d  /  y  /  z  within  "
          f"3  m  /  0.6  m  /  0.6  m  )  ,  {len(bags)}  bags")
    used = [False] * len(all_rows)
    groups = []
    for i, r in enumerate(all_rows):
        if used[i]:
            continue
        used[i] = True
        grp = [r]
        for j in range(i + 1, len(all_rows)):
            if used[j]:
                continue
            s = all_rows[j]
            if (abs(r["d2_m"] - s["d2_m"]) <= 3 and
                    abs(r["y_center_m"] - s["y_center_m"]) <= 0.6 and
                    abs(r["z_center"] - s["z_center"]) <= 0.6):
                grp.append(s)
                used[j] = True
        groups.append(grp)
    groups.sort(key=lambda g: -len(g))
    multi = [g for g in groups if len(g) >= 2]
    if multi:
        print("    boxes  seen  in  several  bags  :")
    for gi, g in enumerate(multi):
        bl = ",  ".join(sorted(set(x["bag"].split('_')[0] for x in g)))
        d = float(np.median([x["d2_m"] for x in g]))
        w = float(np.median([x["width_m"] for x in g]))
        h = float(np.median([x["height"] for x in g]))
        yc = float(np.median([x["y_center_m"] for x in g]))
        print(f"      group  {gi}  :  d  =  {d:.1f}  m  ,  y  =  "
              f"{yc:.1f}  m  ,  {w:.2f}  x  {h:.2f}  m  ,  {len(g)}  "
              f"detections  ({bl})")
    single = sum(1 for g in groups if len(g) == 1)
    print(f"    single-bag  boxes  :  {single}")


def main():
    ap = argparse.ArgumentParser(
        description="detect  synthetic  boxes  from  the  shadow  "
                    "screens  (  angular  step  analysis  )",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--gap", type=float, default=None,
                    help="screen  exclusion  depth  (  default  =  step  )")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--dmax", type=float, default=None)
    ap.add_argument("--yspan", type=float, default=None)
    ap.add_argument("--zmin", type=float, default=None)
    ap.add_argument("--zmax", type=float, default=None)
    ap.add_argument("--bin", dest="bin_m", type=float, default=0.05)
    ap.add_argument("--z-scale", type=float, default=1.0,
                    help="scale  for  the  0..255  slot  (  0.01  for  "
                         "the  hackathon  data  )")
    ap.add_argument("--forward-x", action="store_true")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--per-frame", type=int, default=50_000)
    ap.add_argument("--ang-half", type=float, default=0.12,
                    help="angular  map  half  -  width  a  =  y  /  f  "
                         "(  rad  )")
    ap.add_argument("--bmin", type=float, default=-0.16,
                    help="b  =  z  /  f  lower  bound  (  rad  )")
    ap.add_argument("--bmax", type=float, default=0.30,
                    help="b  =  z  /  f  upper  bound  (  rad  )")
    ap.add_argument("--min-d", type=float, default=15.0,
                    help="reject  boxes  closer  than  this  (  vehicle  "
                         "zone  )")
    ap.add_argument("--max-size", type=float, default=8.0,
                    help="reject  wider  /  taller  than  this  (  m  )")
    ap.add_argument("--min-pts", type=int, default=30)
    ap.add_argument("--min-cells", type=int, default=12,
                    help="min  drop  cells  for  a  box  cluster  ")
    ap.add_argument("--merge-w", type=int, default=4,
                    help="cells  ,  bridge  gaps  when  clustering  "
                         "drop  cells  ")
    ap.add_argument("--t-peak", type=int, default=3,
                    help="per-cell  peak  count  threshold  ")
    ap.add_argument("--t-frac", type=float, default=0.05,
                    help="late  /  peak  ratio  for  a  candidate  cell ")
    ap.add_argument("--late-n", type=int, default=3,
                    help="number  of  last  screens  counted  as  late ")
    ap.add_argument("--ramp-win", type=int, default=6,
                    help="screens  looked  back  for  the  ramp  test ")
    ap.add_argument("--ramp-ratio", type=float, default=1.5,
                    help="ramp  rejection  ratio  ")
    ap.add_argument("--no-annotate", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "shadows"))
    a = ap.parse_args()

    all_rows = []
    for fp_ in a.files:
        all_rows += process(fp_, a, a.outdir)
    cross_bag(all_rows)
    print("done")


if __name__ == "__main__":
    main()
