#!/usr/bin/env python3
"""
_self-test  for  lidar_slices.py  :

a  synthetic  corridor  (  floor  z  =  -1.075  m  ,  walls  y  =  ±3  ,
150  m  long  ,  sensor  at  the  origin  )  with  a  2x2x2  m  box  on
the  floor  at  f  =  45..47  m  ,  y  =  0.5..2.5  m  .  Background
points  hidden  behind  the  box  (  segment  -  AABB  slab  test  from
the  origin  )  are  removed  -  exactly  what  a  real  LiDAR  does  .

Expected  :
  *  screens  with  D  +  gap  <  45  :  the  box  shows  as  a  BRIGHT
    BLOB  at  (  y  *  D  /  f  ,  z  *  D  /  f  )  ;
  *  screens  with  D  +  gap  >=  47  :  the  box  is  excluded  and  a
    DARK  RECTANGLE  (  its  shadow  )  appears  at  the  angular
    footprint  of  the  box  ,  magnified  by  D  /  47  .

Writes  las_out  /  _synthetic.las  and  checks  the  grids  numerically
.
"""
import os
import sys

import numpy as np
import laspy

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

FBOX_LO, FBOX_HI = 45.0, 47.0
YBOX_LO, YBOX_HI = 0.5, 2.5
ZBOX_LO, ZBOX_HI = -1.075, 0.925


def seg_hits_box(P, box):
    """True  where  the  segment  origin  ->  P  crosses  the  box
    (  slab  test  ,  vectorised  )  .  P  :  (  N  ,  3  )  in  (  f  ,
    y  ,  z  )  ."""
    tmin = np.full(len(P), -np.inf)
    tmax = np.full(len(P), np.inf)
    for a in range(3):
        pa = P[:, a]
        lo, hi = box[a]
        nz = pa != 0
        r1 = np.where(nz, lo / np.where(nz, pa, 1.0), -np.inf)
        r2 = np.where(nz, hi / np.where(nz, pa, 1.0), np.inf)
        tmin = np.maximum(tmin, np.minimum(r1, r2))
        tmax = np.minimum(tmax, np.maximum(r1, r2))
    eps = 1e-6
    return (tmax > tmin + eps) & (tmin < 1 - eps) & (tmax > eps)


#  -------------------------------------------------------------  scene
print("building  synthetic  corridor  +  box  ...")
pts = []
ff = np.arange(2, 151, 0.2)
yy = np.linspace(-2.9, 2.9, 26)
F, Y = np.meshgrid(ff, yy, indexing="ij")
pts.append(np.column_stack([F.ravel(), Y.ravel(),
                             np.full(F.size, -1.075)]))
zz = np.linspace(-1.075, 2.5, 14)
for sgn in (-1, 1):
    Fw, Zw = np.meshgrid(ff, zz, indexing="ij")
    pts.append(np.column_stack([Fw.ravel(),
                                 np.full(Fw.size, sgn * 3.0),
                                 Zw.ravel()]))
bg = np.vstack(pts)
box3 = ((FBOX_LO, FBOX_HI), (YBOX_LO, YBOX_HI), (ZBOX_LO, ZBOX_HI))
hit = seg_hits_box(bg, box3)
print(f"  background  :  {len(bg):d}  pts  ,  occluded  by  the  box  :  {int(hit.sum()):d}  (  {100 * hit.mean():.1f}  %  )")
bg = bg[~hit]

#  box  faces  (  front  ,  back  ,  top  ,  two  sides  )
rows = []
for _ in range(200):
    rows.append((FBOX_LO, rng.uniform(YBOX_LO, YBOX_HI),
                 rng.uniform(ZBOX_LO, ZBOX_HI)))
    rows.append((FBOX_HI, rng.uniform(YBOX_LO, YBOX_HI),
                 rng.uniform(ZBOX_LO, ZBOX_HI)))
    rows.append((rng.uniform(FBOX_LO, FBOX_HI),
                 rng.uniform(YBOX_LO, YBOX_HI), ZBOX_HI))
    rows.append((rng.uniform(FBOX_LO, FBOX_HI), YBOX_LO,
                 rng.uniform(ZBOX_LO, ZBOX_HI)))
    rows.append((rng.uniform(FBOX_LO, FBOX_HI), YBOX_HI,
                 rng.uniform(ZBOX_LO, ZBOX_HI)))
box_pts = np.array(rows)

allx = np.concatenate([-bg[:, 0], -box_pts[:, 0]])
ally = np.concatenate([bg[:, 1], box_pts[:, 1]])
allz = np.concatenate([bg[:, 2], box_pts[:, 2]])
print(f"  box  points  :  {len(box_pts):d}   (  total  {len(allx)}  )")

#  --------------------------------------------------------------  .las
out = os.path.join(HERE, "las_out", "_synthetic.las")
L = laspy.create(point_format=3)
L.header.x_scale = L.header.y_scale = L.header.z_scale = 0.001
L.header.x_offset = L.header.y_offset = L.header.z_offset = 0.0
L.header.x_max = int(np.round(allx.max() * 1000))
L.header.y_max = int(np.round(ally.max() * 1000))
L.header.z_max = int(np.round(allz.max() * 1000))
L.header.x_min = int(np.round(allx.min() * 1000))
L.header.y_min = int(np.round(ally.min() * 1000))
L.header.z_min = int(np.round(allz.min() * 1000))
L.header.point_count = len(allx)
L.x = allx.astype(np.float64)
L.y = ally.astype(np.float64)
L.z = allz.astype(np.float64)
L.intensity = np.zeros(len(allx), np.uint16)
L.return_number = np.zeros(len(allx), np.uint8)
L.flags = np.zeros(len(allx), np.uint8)
L.scan_angle_rank = np.zeros(len(allx), np.int16)
L.user_data = np.zeros(len(allx), np.uint8)
L.scan_direction_flag = np.zeros(len(allx), np.uint8)
L.edge_of_flight = np.zeros(len(allx), np.uint8)
L.write(out)
print(f"  wrote  {out}")

#  -------------------------------------------------------------  check
sys.path.insert(0, HERE)
import lidar_slices as ls

x, y, z = ls.load_las(out)
f = -x
print(f"  read  back  :  {len(x)}  pts  ,  f  =  "
      f"{f.min():.1f}  ..  {f.max():.1f}  m  ,  z  =  "
      f"{z.min():.3f}  ..  {z.max():.3f}  m")

step, gap = 5.0, 5.0
YS, ZMIN, ZMAX, BIN = 4.0, -2.0, 3.0, 0.05
D_list, grids, npts = ls.build_slices(
    f, y, z, step, gap, step, 70.0, YS, ZMIN, ZMAX, BIN)
n_y = grids[D_list[0]].shape[1]
n_z = grids[D_list[0]].shape[0]
bin_y = 2 * YS / n_y
bin_z = (ZMAX - ZMIN) / n_z


def region(g, u0, u1, v0, v1):
    i0 = int((u0 + YS) / bin_y + 1e-6)
    i1 = int((u1 + YS) / bin_y + 1e-6)
    j0 = int((v0 - ZMIN) / bin_z + 1e-6)
    j1 = int((v1 - ZMIN) / bin_z + 1e-6)
    return int(g[max(0, j0):j1, max(0, i0):i1].sum())


#  expected  number  of  box  points  inside  the  (  shrunk  )
#  footprint  of  each  screen  -  computed  directly  from  the
#  box  geometry  .
bf, by, bz = box_pts[:, 0], box_pts[:, 1], box_pts[:, 2]

ok = True
for D in D_list:
    g = grids[D]
    u_lo, u_hi = 0.5 * D / FBOX_HI, 2.5 * D / FBOX_HI
    v_lo, v_hi = -1.075 * D / FBOX_HI, 0.925 * D / FBOX_HI
    m = 0.08                       #  stay  inside  the  footprint
    inr = region(g, u_lo + m, u_hi - m, v_lo + m, v_hi - m)
    ref = region(g, u_lo - 3.0 + m, u_hi - 3.0 - m, v_lo + m, v_hi - m)
    q = bf > D + gap
    uq = by[q] * D / bf[q]
    vq = bz[q] * D / bf[q]
    inside = ((uq >= u_lo + m) & (uq < u_hi - m) &
              (vq >= v_lo + m) & (vq < v_hi - m))
    E = int(inside.sum())
    beyond = D + gap >= FBOX_HI
    tag = "shadow  expected" if beyond else "blob  expected"
    if beyond:
        good = inr < max(5, 0.15 * ref)
        extra = ""
    else:
        good = E > 0 and inr >= 0.5 * E
        extra = f"  ,  expected  box  pts  {E}"
    ok &= good
    print(f"  D  =  {D:4.1f}  m  :  {tag}  ;  footprint  inner  =  {inr:6d}  pts{extra}  ,  clear  reference  =  {ref:6d}  pts   ->  {'PASS' if good else 'FAIL'}")

print("\nALL  PASS" if ok else "\nSOME  FAILED")
sys.exit(0 if ok else 1)
