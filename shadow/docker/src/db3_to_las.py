#!/usr/bin/env python3
"""
db3_to_las.py  —  convert  the  hackathon  .db3  LiDAR  bags  into  standard  .las
and  "see  what  happens"  with  the  data  when  a  normal  3D  format  trusts
the  fields  as  declared.

Record  layout  (  26  bytes,  data  from  byte  200  of  the  CDR  blob  )  :
     x  float32  @ 0    —  metres  (  forward  =  -x  )
     y  float32  @ 4    —  metres
     z  float32  @ 8    —  0..255  per-point  attribute  (  NOT  calibrated  )
     intensity  @12     —  producer  garbage  (  -0,  +/-1,  +/-2,  NaN  )
     ring   uint32  @16 —  base  +  0..4  (  Hesai  )  /  0..9  (  Livox  )
     timestamp  @20     —  8  bytes

Mapping  into  LAS  1.2  point  format  2  (  the  naive  "trust  the  fields"
mapping  a  standard  converter  would  do  )  :
     X  =  x  ·  1000      (  LAS  xyz  is  in  mm,  scale  =  0.001  )
     Y  =  y  ·  1000
     Z  =  z_slot  ·  1000  —  i.e.  the  0..255  value  is  stored  as  if  it
                               were  a  METRIC  height  :  0..255  m  !
     intensity  =  clamp  (  intensity_slot  )        (  mostly  0  )
     classification  =  ring  -  min  (  ring  )      (  0..4  /  0..9  )
     point_source_id  =  frame  index

Outputs  (  into  --lasdir  )  :
     <bag>.las
     view_top_<bag>.png      (  x,  y  )  —  the  usual  corridor
     view_side_<bag>.png     (  x,  z  )  —  the  "  255-m  column  "
     view_end_<bag>.png      (  y,  z  )
plus  a  console  report  of  what  the  z  /  intensity  /  channel  statistics
look  like  through  the  LAS  lens.
"""

import os
import sys
import argparse
import sqlite3
import numpy as np
import laspy

HERE = os.path.dirname(os.path.abspath(__file__))
BAG_DIR = os.path.join(HERE, "for_hackathon")

def pc_data(blob):
    raw = np.frombuffer(blob, np.uint8)[200:]
    n = len(raw) // 26
    p = raw[:n * 26].reshape(n, 26)
    x = p[:, 0:4].view(np.float32).reshape(-1)
    y = p[:, 4:8].view(np.float32).reshape(-1)
    z = p[:, 8:12].view(np.float32).reshape(-1)
    i = p[:, 12:16].view(np.float32).reshape(-1)
    rg = p[:, 16:20].view(np.uint32).reshape(-1)
    k = (np.hypot(x, y) > 0.3) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return (x[k], y[k], z[k], i[k], rg[k])


def collect(bag, frame_step, point_step):
    path = os.path.join(BAG_DIR, bag, bag + "_0.db3")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    nmsg = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    xs, ys, zs, its, chs, psids = [], [], [], [], [], []
    for fi in range(0, nmsg, frame_step):
        blob = con.execute(
            "SELECT data FROM messages ORDER BY timestamp LIMIT 1 OFFSET %d" % fi
        ).fetchone()[0]
        x, y, z, i, rg = pc_data(blob)
        j = np.arange(0, len(x), point_step)
        x, y, z, i, rg = x[j], y[j], z[j], i[j], rg[j]
        base = int(rg.min()) if len(rg) else 0
        xs.append(x)
        ys.append(y)
        zs.append(np.round(z))                       #  z  slot  =  the  0..255  attribute
        its.append(i)
        chs.append((rg - base).astype(np.int64))
        psids.append(np.full(len(x), fi, dtype=np.int64))
    con.close()
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs), \
        np.concatenate(its), np.concatenate(chs), np.concatenate(psids)


def to_las(bag, x, y, z, it, ch, psid, out_path):
    las = laspy.create(point_format=2)
    #  scale  MUST  be  set  BEFORE  dimension  assignment  (  laspy  default  is
    #  0.01,  and  assignment  converts  through  the  current  scale  )
    las.header.x_scale = las.header.y_scale = las.header.z_scale = 0.001
    las.header.x_offset = las.header.y_offset = las.header.z_offset = 0.0
    las.x = x.astype(np.float64)
    las.y = y.astype(np.float64)
    las.z = z.astype(np.float64)                    #  0..255  "  metres  "
    las.intensity = np.nan_to_num(it, nan=0.0).clip(0, 65535).astype(np.uint16)
    las.classification = ch.astype(np.uint8)
    las.point_source_id = psid.astype(np.uint16)
    las.header.producer_id = 9999
    las.header.software_id = "db3_to_las"
    las.write(out_path)
    return out_path


def report(bag, las):
    z = np.asarray(las.z, dtype=np.float64)
    it = np.asarray(las.intensity, dtype=np.float64)
    ch = np.asarray(las.classification, dtype=np.int64)
    print(f"\n==  {bag}.las   (  {len(las):,}  points  )")
    print(f"   X  [  m  ]  :  min  {las.x.min():8.2f}   max  {las.x.max():8.2f}")
    print(f"   Y  [  m  ]  :  min  {las.y.min():8.2f}   max  {las.y.max():8.2f}")
    print(f"   Z  [  m  ]  :  min  {z.min():8.2f}   max  {z.max():8.2f}   "
          f"mean  {z.mean():7.2f}   median  {np.median(z):6.1f}")
    u, c = np.unique(z, return_counts=True)
    top = np.argsort(c)[-6:][::-1]
    print(f"   z  top  values  :  "
          f"{[(int(u[t]), int(c[t])) for t in top]}")
    u2, c2 = np.unique(it, return_counts=True)
    print(f"   intensity  :  distinct  =  {len(u2)},  top  :  "
          f"{[(float(u2[t]), int(c2[t])) for t in np.argsort(c2)[-4:][::-1]]}")
    print("   per  channel  (  ring  -  min  )  :  n  ,  z  p10/50/90  ,  z  range")
    for cch in np.unique(ch):
        m = ch == cch
        print(f"     ch  {cch}:  n  =  {m.sum():7d}   "
              f"z  =  {np.percentile(z[m], [10, 50, 90]).round(1)}   "
              f"[{z[m].min():.0f}..{z[m].max():.0f}]  m")


def view(hname, vname, h, v, tag, title, xlim=None, ylim=None, nmax=200000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(h)
    if n > nmax:
        j = np.random.default_rng(0).choice(n, nmax, replace=False)
        h, v = h[j], v[j]
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="black")
    ax.set_facecolor("black")
    ax.scatter(h, v, s=0.25, c="white", linewidths=0)
    ax.set_title(title, color="white")
    ax.set_xlabel(hname, color="white")
    ax.set_ylabel(vname, color="white")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("gray")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    out = os.path.join(OUT, f"view_{tag}_{BAG}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"   ->  {out}")


OUT = None
BAG = None


def main():
    global OUT, BAG
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="*",
                    default=["doubleT_platform", "doubleT_obstacle"])
    ap.add_argument("--frame-step", type=int, default=5)
    ap.add_argument("--point-step", type=int, default=20)
    ap.add_argument("--lasdir", default=os.path.join(HERE, "las_out"))
    a = ap.parse_args()
    OUT = a.lasdir
    os.makedirs(OUT, exist_ok=True)
    for bag in a.bags:
        BAG = bag
        print(f"==  collecting  {bag}  (  frame  step  {a.frame_step},  "
              f"point  step  {a.point_step}  )  ...")
        x, y, z, it, ch, psid = collect(bag, a.frame_step, a.point_step)
        print(f"   points  :  {len(x):,}")
        p = to_las(bag, x, y, z, it, ch, psid,
                   os.path.join(OUT, bag + ".las"))
        print(f"   wrote   :  {p}  ({os.path.getsize(p) / 1e6:.1f}  MB  )")
        las = laspy.read(p)
        report(bag, las)
        view("x  [  m  ]", "y  [  m  ]", x, y, "top",
             f"{bag}  —  top  view  (  x,  y  )  from  .las",
             xlim=(min(x), max(x)))
        view("x  [  m  ]", "z  [  m  ]  (  slot  0..255  as  metres  )",
             x, z, "side",
             f"{bag}  —  side  view  (  x,  z  )  :  the  255-m  phantom  column",
             xlim=(min(x), max(x)), ylim=(-5, 260))
        view("y  [  m  ]", "z  [  m  ]  (  slot  0..255  as  metres  )",
             y, z, "end",
             f"{bag}  —  end  view  (  y,  z  )  from  .las",
             xlim=(min(y), max(y)), ylim=(-5, 260))


if __name__ == "__main__":
    main()
