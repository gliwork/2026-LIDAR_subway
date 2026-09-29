#!/usr/bin/env python3
"""
far_points.py - show exactly what the 170-200 m returns are made of.

For a bag, takes all points with forward distance in [d_lo, d_hi]
(default 170-200 m) over several frames, clusters them (3D grid
clustering, eps default 1.5 m - far returns are sparse), classifies
each cluster against the tunnel model (left wall / right wall / floor
/ ceiling / exit-other), and prints a per-cluster table plus top and
side ASCII views, and saves CSV + PNG per scenario.

Usage:
    source /opt/ros/humble/setup.bash
    /usr/bin/python3.10 far_points.py for_hackathon/<scenario> \
        [--frames 16] [--dlo 170] [--dhi 200] [--eps 1.5]
"""
import argparse
import csv
import os
import re

import numpy as np

from slice_detect import open_reader, parse_points
from tunnel_model import build_model, normals, SURF, NAMES


def grid_clusters(x, d_, z, eps, min_pts):
    """simple 3D grid clustering; returns list of index-masks."""
    n = len(x)
    if n < min_pts:
        return []
    cell = np.stack([
        ((x - x.min()) / eps).astype(int),
        ((d_ - d_.min()) / eps).astype(int),
        ((z - z.min()) / eps).astype(int),
    ], axis=1)
    key = cell[:, 0] * 1000000 + cell[:, 1] * 1000 + cell[:, 2]
    order = np.argsort(key)
    parents = list(range(n))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parents[max(ri, rj)] = min(ri, rj)

    seen = {}
    for i in order:
        k = key[i]
        ci = cell[i]
        if k in seen:
            union(i, seen[k])
        for dx_ in (-1, 0, 1):
            for dy_ in (-1, 0, 1):
                for dz_ in (-1, 0, 1):
                    kk = ((ci[0] + dx_) * 1000000
                          + (ci[1] + dy_) * 1000 + (ci[2] + dz_))
                    if kk in seen:
                        union(i, seen[kk])
        seen[k] = i

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.array(g) for g in groups.values() if len(g) >= min_pts]


def classify(cx, dz_, cz, dgrid, model, nrm, tol=0.9):
    j = int(np.clip((dz_ - dgrid[0]) / (dgrid[1] - dgrid[0]), 0,
                    len(dgrid) - 1))
    cands = []
    if cx < 0 and abs(cx - model["x_left"][j]) < tol:
        cands.append("left wall")
    if cx > 0 and abs(cx - model["x_right"][j]) < tol:
        cands.append("right wall")
    if abs(cz - model["z_floor"][j]) < tol:
        cands.append("floor")
    if abs(cz - model["z_ceil"][j]) < tol:
        cands.append("ceiling")
    if cands:
        return " + ".join(cands)
    # distance to the tunnel envelope -> probably outside/exit
    dxw = min(abs(cx - model["x_left"][j]), abs(model["x_right"][j] - cx))
    dzs = min(abs(cz - model["z_floor"][j]), abs(cz - model["z_ceil"][j]))
    return f"other (d_wall {dxw:.1f}, d_surf {dzs:.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--dlo", type=float, default=170.0)
    ap.add_argument("--dhi", type=float, default=200.0)
    ap.add_argument("--eps", type=float, default=1.5)
    ap.add_argument("--min-pts", type=int, default=3)
    ap.add_argument("--dmax", type=float, default=215.0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir
    if outdir is None:
        m = re.match(r"(.*for_hackathon/)(\w+)", args.bag)
        outdir = (m.group(1) + "../out/" + m.group(2)) if m else "."
    os.makedirs(outdir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.bag))

    dgrid, model = build_model(args.bag, args.frames, args.dmax)
    nrm = normals(dgrid, model)

    r = open_reader(args.bag)
    f = 0
    pts = []
    while r.has_next() and f < args.frames:
        _, d, _ = r.read_next()
        x, y, z, inten = parse_points(d)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z, inten = x[m], y[m], z[m], inten[m]
        df = -y
        sel = (df >= args.dlo) & (df <= args.dhi)
        if sel.sum():
            pts.append((x[sel], df[sel], z[sel], inten[sel]))
        f += 1

    if not pts:
        print(f"[far_points] {tag}: no points in {args.dlo:.0f}-{args.dhi:.0f} m")
        return
    x = np.concatenate([p[0] for p in pts])
    df = np.concatenate([p[1] for p in pts])
    z = np.concatenate([p[2] for p in pts])
    inten = np.concatenate([p[3] for p in pts])
    print(f"[far_points] {tag}: {len(x)} points in "
          f"{args.dlo:.0f}-{args.dhi:.0f} m over {len(pts)} frames")

    clus = grid_clusters(x, df, z, args.eps, args.min_pts)
    clus.sort(key=lambda g: -len(g))

    rows = []
    for g in clus:
        cx, cd, cz = float(x[g].mean()), float(df[g].mean()), float(z[g].mean())
        rows.append({
            "n": int(len(g)),
            "cx": cx, "cd": cd, "cz": cz,
            "dx": float(x[g].max() - x[g].min()),
            "dd": float(df[g].max() - df[g].min()),
            "dz": float(z[g].max() - z[g].min()),
            "i": float(inten[g].mean()),
            "what": classify(cx, cd, cz, dgrid, model, nrm),
        })

    with open(os.path.join(outdir, "far_clusters.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "cx", "cd", "cz", "span_x", "span_d", "span_z",
                    "intensity", "classification"])
        for r_ in rows:
            w.writerow([r_["n"], f"{r_['cx']:.2f}", f"{r_['cd']:.1f}",
                        f"{r_['cz']:.2f}", f"{r_['dx']:.2f}",
                        f"{r_['dd']:.2f}", f"{r_['dz']:.2f}",
                        f"{r_['i']:.0f}", r_["what"]])

    tot_c = sum(r_["n"] for r_ in rows)
    print(f"   {len(rows)} clusters, {tot_c} pts clustered "
          f"({100 * tot_c / len(x):.0f}% of far points)\n")
    print(f"{'n':>5} {'x':>6} {'d':>6} {'z':>6}  {'span x/d/z':>14}  "
          f"{'int':>4}  classification")
    for r_ in rows[:25]:
        print(f"{r_['n']:>5} {r_['cx']:>+6.2f} {r_['cd']:>6.1f} "
              f"{r_['cz']:>6.2f}  "
              f"{r_['dx']:.1f}/{r_['dd']:.1f}/{r_['dz']:.1f}    "
              f"{r_['i']:>4.0f}  {r_['what']}")
    if len(rows) > 25:
        print(f"   ... +{len(rows) - 25} more in far_clusters.csv")

    # ---- ASCII views (top: x vs d ; side: z vs d) ---------------------
    def ascii_map(vals_a, vals_b, label_a, label_b, x0, x1, y0, y1,
                  nxb, nyb, marks=()):
        grid = np.full((nyb, nxb), " ")
        counts = np.zeros((nyb, nxb), dtype=int)
        for a, b in zip(vals_a, vals_b):
            i = int((a - x0) / (x1 - x0) * (nxb - 1))
            j = int((b - y0) / (y1 - y0) * (nyb - 1))
            i = min(max(i, 0), nxb - 1)
            j = min(max(j, 0), nyb - 1)
            grid[j, i] = "#"
            counts[j, i] += 1
        for (a, b, ch) in marks:
            i = int((a - x0) / (x1 - x0) * (nxb - 1))
            j = int((b - y0) / (y1 - y0) * (nyb - 1))
            i = min(max(i, 0), nxb - 1)
            j = min(max(j, 0), nyb - 1)
            grid[j, i] = ch
        lines = []
        for j in range(nyb - 1, -1, -1):
            row = "".join(grid[j])
            row = re.sub(r"#+", lambda m: "." if len(m.group()) < 4
                         else ("o" if len(m.group()) < 8 else "#"), row)
            yv = y0 + (nyb - 1 - j) / (nyb - 1) * (y1 - y0)
            lines.append(f"{yv:5.0f} |{row}")
        xa = [f"{x0 + i / (nxb - 1) * (x1 - x0):>5.0f}" for i in
              range(0, nxb, max(1, nxb // 8))]
        lines.append("      +" + "-" * nxb)
        lines.append("       " + " ".join(xa))
        lines.append(f"       {label_a} (m)   [{label_b} vs {label_a}]"
                     )
        return "\n".join(lines)

    marks = [(r_["cx"], r_["cd"], "*") for r_ in rows if r_["n"] >= 20]
    print("\nTOP (x vs d)  '#'=dense  'o'=medium  '.'=sparse  '*'=big cluster")
    print(ascii_map(df, x, "d", "x", args.dlo - 5, args.dhi + 5, -10, 10,
                    76, 24, marks))
    print("\nSIDE (z vs d)")
    print(ascii_map(df, z, "d", "z", args.dlo - 5, args.dhi + 5, -4, 6,
                    76, 18,
                    [(r_["cd"], r_["cz"], "*") for r_ in rows
                     if r_["n"] >= 20]))

    # ---- PNG ----------------------------------------------------------
    try:
        from PIL import Image, ImageDraw, ImageFont

        def mkf(x0, x1, y0, y1, xr, yr):
            def f(px, py):
                fx = (px - xr[0]) / (xr[1] - xr[0])
                fy = (py - yr[0]) / (yr[1] - yr[0])
                return (x0 + fx * (x1 - x0), y1 - fy * (y1 - y0))
            return f

        img = Image.new("RGB", (1500, 900), "white")
        drw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        f0 = mkf(20, 740, 30, 380, (args.dlo - 5, args.dhi + 5), (-10, 10))
        f1 = mkf(20, 740, 430, 780, (args.dlo - 5, args.dhi + 5), (-4, 6))
        f2 = mkf(780, 1480, 30, 780, (args.dlo - 5, args.dhi + 5),
                 (-4, 10))
        drw.rectangle([20, 30, 740, 380], outline="black", width=1)
        drw.text((25, 35), f"{tag} TOP (d vs x)", fill="black", font=font)
        drw.rectangle([20, 430, 740, 780], outline="black", width=1)
        drw.text((25, 435), f"{tag} SIDE (d vs z)", fill="black", font=font)
        drw.rectangle([780, 30, 1480, 780], outline="black", width=1)
        drw.text((785, 35), f"{tag} SIDE, wide z", fill="black", font=font)

        # model curves
        m0 = (model["x_left"], (0, 0, 255))
        m1 = (model["x_right"], (255, 140, 0))
        for d0 in range(int(args.dlo - 5), int(args.dhi + 5), 2):
            j = int(np.clip((d0 - dgrid[0]) / (dgrid[1] - dgrid[0]), 0,
                            len(dgrid) - 1))
            drw.line([f0(d0, model["x_left"][j]),
                      f0(d0 + 2, model["x_left"][j + 1]
                         if j + 1 < len(dgrid) else model["x_left"][j])],
                     fill=(0, 0, 255), width=2)
            drw.line([f0(d0, model["x_right"][j]),
                      f0(d0 + 2, model["x_right"][j + 1]
                         if j + 1 < len(dgrid) else model["x_right"][j])],
                     fill=(255, 140, 0), width=2)
            drw.line([f1(d0, model["z_floor"][j]),
                      f1(d0 + 2, model["z_floor"][j + 1]
                         if j + 1 < len(dgrid) else model["z_floor"][j])],
                     fill=(0, 160, 0), width=2)
            drw.line([f1(d0, model["z_ceil"][j]),
                      f1(d0 + 2, model["z_ceil"][j + 1]
                         if j + 1 < len(dgrid) else model["z_ceil"][j])],
                     fill=(200, 0, 200), width=2)

        # scatter (subsample)
        st = max(1, len(x) // 4000)
        for i in range(0, len(x), st):
            drw.point(f0(df[i], x[i]), fill=(200, 200, 200))
            drw.point(f1(df[i], z[i]), fill=(200, 200, 200))
        # clusters
        colmap = {"left wall": (255, 0, 0), "right wall": (0, 120, 0),
                  "floor": (255, 140, 0), "ceiling": (200, 0, 200)}
        for r_ in rows:
            c = (128, 128, 128)
            for key, cc in colmap.items():
                if r_["what"].startswith(key):
                    c = cc
                    break
            x0_, x1_ = r_["cx"] - r_["dx"] / 2, r_["cx"] + r_["dx"] / 2
            d0_, d1_ = r_["cd"] - r_["dd"] / 2, r_["cd"] + r_["dd"] / 2
            drw.rectangle([f0(d0_, x0_), f0(d1_, x1_)], outline=c, width=2)
            z0_, z1_ = r_["cz"] - r_["dz"] / 2, r_["cz"] + r_["dz"] / 2
            drw.rectangle([f1(d0_, z0_), f1(d1_, z1_)], outline=c, width=2)
            if r_["n"] >= 20:
                drw.text(f0(r_["cd"], r_["cx"]),
                         f"{r_['n']}", fill=c, font=font)

        drw.text((785, 790),
                 "grey = raw far points; red=left wall, green=right wall, "
                 "orange=floor, magenta=ceiling, grey box=other/exit; "
                 "blue/orange solid = modelled walls; "
                 "green/magenta = modelled floor/ceiling",
                 fill="black", font=font)
        p = os.path.join(outdir, "far_points.png")
        img.save(p)
        print(f"\n[far_points] wrote {p} and "
              f"{os.path.join(outdir, 'far_clusters.csv')}")
    except Exception as e:
        print(f"[far_points] plot failed: {e}")


if __name__ == "__main__":
    main()
