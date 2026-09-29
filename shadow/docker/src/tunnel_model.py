#!/usr/bin/env python3
"""
tunnel_model.py - model the tunnel walls / floor / ceiling from a bag and
compute the optimal aim angles for a separate (steerable) laser beam.

Builds, from the pointcloud of one bag:
  x_left(d), x_right(d)     wall lateral position vs forward distance d
  z_floor(d), z_ceiling(d)  (robust percentiles per 2 m bin, smoothed)

Surface normals follow from the derivatives (walls vertical, floor/
ceiling horizontal to first order).  A steerable beam (azimuth a from
forward, elevation e, originating at the lidar) is ray-traced against
the model; for each surface the script searches the aim angles that hit
that surface and minimise the incidence angle - perpendicular hits give
the strongest returns and the densest point clusters.

Outputs (out/<scenario>/):
  tunnel_model.csv   d, x_left, x_right, z_floor, z_ceil, wall normals
  tunnel_model.png   top view + side view with the recommended beams
  (console)          recommended aim per surface, and the incidence
                     angles of the ACTUAL far-range points

Usage:
    source /opt/ros/humble/setup.bash
    /usr/bin/python3.10 tunnel_model.py for_hackathon/<scenario> \
        [--frames 16] [--dmax 210] [--min-hit 10]
"""
import argparse
import csv
import os
import re

import numpy as np

from slice_detect import open_reader, parse_points
from PIL import Image, ImageDraw, ImageFont

SURF = ("x_left", "x_right", "z_floor", "z_ceil")
NAMES = {"x_left": "left wall", "x_right": "right wall",
         "z_floor": "floor", "z_ceil": "ceiling"}


# ---------------------------------------------------------------- model
def build_model(bag, n_frames, d_max):
    bins = 2.0
    nb = int(d_max / bins)
    acc = {k: [] for k in SURF}

    r = open_reader(bag)
    f = 0
    while r.has_next() and f < n_frames:
        _, d, _ = r.read_next()
        x, y, z, _ = parse_points(d)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[m], y[m], z[m]
        df = -y
        sel = (df >= 1.5) & (df <= d_max)
        b = np.clip(((df[sel] - 1.5) // bins).astype(int), 0, nb - 1)
        xs, zs = x[sel], z[sel]
        for i in range(nb):
            bi = b == i
            if bi.sum() < 20:
                continue
            w = bi & (np.abs(xs) > 1.6)          # outside the corridor
            xl = xs[w & (xs < 0)]
            xr = xs[w & (xs > 0)]
            if len(xl) > 5:
                acc["x_left"].append((i, np.percentile(xl, 3)))
            if len(xr) > 5:
                acc["x_right"].append((i, np.percentile(xr, 97)))
            zf = np.percentile(zs[bi], 3)
            zc = np.percentile(zs[bi], 97)
            if -3.0 < zf < 0.6:
                acc["z_floor"].append((i, zf))
            if 0.5 < zc < 6.0:
                acc["z_ceil"].append((i, zc))
        f += 1

    dgrid = np.arange(1.5, d_max, bins)
    model = {}
    for key in SURF:
        v = np.full(nb, np.nan)
        for i, val in acc[key]:
            v[i] = val
        nan = np.isnan(v)
        if nan.any() and (~nan).sum() > 4:
            v[nan] = np.interp(dgrid[nan], dgrid[~nan], v[~nan])
        k = 5                                    # 11-bin median smooth
        for i in range(nb):
            v[i] = np.median(v[max(0, i - k):i + k + 1])
        model[key] = v
    return dgrid, model


def normals(dgrid, model):
    """3D unit surface normals per d sample."""
    out = {}
    xl, xr = model["x_left"], model["x_right"]
    dxl = np.gradient(xl, dgrid)
    dxr = np.gradient(xr, dgrid)
    nl = np.stack([np.ones_like(dxl), dxl, np.zeros_like(dxl)], axis=1)
    nr = np.stack([-np.ones_like(dxr), -dxr, np.zeros_like(dxr)], axis=1)
    nf = np.stack([np.zeros_like(dxl),
                   np.gradient(model["z_floor"], dgrid),
                   np.ones_like(dxl)], axis=1)
    nc = np.stack([np.zeros_like(dxl),
                   np.gradient(model["z_ceil"], dgrid),
                   -np.ones_like(dxl)], axis=1)
    for name, n in (("x_left", nl), ("x_right", nr),
                    ("z_floor", nf), ("z_ceil", nc)):
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        out[name] = n
    return out


# ------------------------------------------------------------- ray trace
def trace_beams(dgrid, model, az, el, d_min, d_max):
    """First tunnel-surface hit + incidence angle for every aim.

    Vectorised: for each azimuth, all elevations are traced together on
    a dense t grid."""
    res = {}
    t = np.arange(0.25, 301, 0.25)[None, :]          # (1, nt)
    ntc = t.shape[1] - 1
    azr = np.radians(az)
    elr = np.radians(el)
    for kind in SURF:
        curve = model[kind]
        out = np.full((len(az), len(el), 4), np.nan)
        for ia, a in enumerate(azr):
            dx = np.sin(a) * np.cos(elr)[:, None]     # (ne, 1)
            dy = -np.cos(a) * np.cos(elr)[:, None]
            dzv = np.sin(elr)[:, None]
            xr = t * dx
            drf = -t * dy
            zr = t * dzv
            fcurve = np.interp(drf.ravel(), dgrid, curve).reshape(
                el.size, t.shape[1])
            ok2 = drf[:, 1:] > 0
            if kind.startswith("x"):
                f = xr - fcurve
                msk = ok2 & (f[:, 1:] * f[:, :-1] <= 0) & \
                      (f[:, :-1] != 0)
            else:
                f = zr - fcurve
                if kind == "z_floor":
                    msk = ok2 & (f[:, :-1] > 0) & (f[:, 1:] <= 0)
                else:
                    msk = ok2 & (f[:, :-1] < 0) & (f[:, 1:] >= 0)
            has = msk.any(axis=1)
            if not has.any():
                continue
            fl = np.fliplr(msk)
            first = np.argmax(fl, axis=1)            # 0 if none -> fix
            first = np.where(has, ntc - 1 - first, 0)
            for ie in np.where(has)[0]:
                i0 = int(first[ie]) + 1
                th = 0.5 * (t[0, i0] + t[0, i0 - 1])
                dh = -th * dy[ie, 0]
                xh = th * dx[ie, 0]
                zh = th * dzv[ie, 0]
                if not (d_min <= dh <= d_max):
                    continue
                j = int(np.clip((dh - dgrid[0]) / (dgrid[1] - dgrid[0]),
                                0, len(dgrid) - 1))
                if kind.startswith("x"):
                    # wall hit must be between floor and ceiling
                    if not (model["z_floor"][j] - 0.5 <= zh
                            <= model["z_ceil"][j] + 0.5):
                        continue
                else:
                    # floor/ceiling hit must be inside the tunnel width
                    if not (abs(xh) < min(-model["x_left"][j],
                                          model["x_right"][j]) - 0.3):
                        continue
                out[ia, ie] = (dh, xh, zh, th)
        res[kind] = out
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--dmax", type=float, default=210.0)
    ap.add_argument("--min-hit", type=float, default=10.0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir
    if outdir is None:
        m = re.match(r"(.*for_hackathon/)(\w+)", args.bag)
        outdir = (m.group(1) + "../out/" + m.group(2)) if m else "."
    os.makedirs(outdir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.bag))

    print(f"[tunnel_model] {tag}: fitting walls/floor/ceiling "
          f"(frames={args.frames}, dmax={args.dmax:.0f} m)")
    dgrid, model = build_model(args.bag, args.frames, args.dmax)
    dmid = dgrid + (dgrid[1] - dgrid[0]) / 2
    nrm = normals(dgrid, model)

    with open(os.path.join(outdir, "tunnel_model.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["d", "x_left", "x_right", "z_floor", "z_ceil",
                    "nx_left", "ny_left", "nx_right", "ny_right"])
        for i in range(len(dgrid)):
            w.writerow([f"{dmid[i]:.1f}", f"{model['x_left'][i]:.3f}",
                        f"{model['x_right'][i]:.3f}",
                        f"{model['z_floor'][i]:.3f}",
                        f"{model['z_ceil'][i]:.3f}",
                        f"{nrm['x_left'][i,0]:.3f}",
                        f"{nrm['x_left'][i,1]:.3f}",
                        f"{nrm['x_right'][i,0]:.3f}",
                        f"{nrm['x_right'][i,1]:.3f}"])

    print("[tunnel_model] searching optimal beam aims...")
    az = np.arange(-90, 90.1, 0.5)
    el = np.arange(-30, 60.1, 0.5)
    hits = trace_beams(dgrid, model, az, el, args.min_hit, args.dmax)

    print(f"{'surface':<12}{'azim':>6}{'elev':>6}{'hit d':>8}"
          f"{'hit x':>9}{'hit z':>8}{'incid':>8}   note")
    best = {}
    for kind in SURF:
        out = hits[kind]
        valid = np.isfinite(out[..., 0])
        if not valid.any():
            print(f"{NAMES[kind]:<12}  (no hit)")
            continue
        incs = np.full(valid.shape, np.inf)
        for ia in range(len(az)):
            for ie in range(len(el)):
                if not valid[ia, ie]:
                    continue
                dh, xh, zh, th = out[ia, ie]
                j = int(np.clip((dh - dgrid[0]) / (dgrid[1] - dgrid[0]),
                                0, len(dgrid) - 1))
                dirv = np.array([xh, -dh, zh])
                dl = np.linalg.norm(dirv)
                incs[ia, ie] = np.degrees(np.arccos(np.clip(
                    abs(dirv @ nrm[kind][j]) / dl, 0, 1)))
        k = np.unravel_index(incs.argmin(), incs.shape)
        dh, xh, zh, th = out[k]
        best[kind] = (az[k[0]], el[k[1]], dh, xh, zh, incs[k])
        inc = incs[k]
        if kind.startswith("x"):
            note = ("perpendicular hit" if inc < 30
                    else "best available - in a straight tunnel the far "
                         "wall normal points sideways, so a beam from the "
                         "centre is grazing (perpendicular only ~beside "
                         "the vehicle: az ~ +/90, hits at 3-4 m)")
        else:
            note = ("flat surface: perpendicular only close in; beyond "
                    "~1.3/tan(inc) m it is grazing for ANY aim - a "
                    "steerable beam cannot fix flat-floor/ceiling "
                    "density at distance")
        print(f"{NAMES[kind]:<12}{az[k[0]]:>+6.1f}{el[k[1]]:>+6.1f}"
              f"{dh:>8.1f}{xh:>9.2f}{zh:>8.2f}{inc:>8.1f}   {note}")

    # ---- best aim for hitting each surface AT a chosen distance ------
    print()
    print("best beam to hit each surface at a target distance "
          "(az = azimuth from forward, el = elevation up; "
          "inc = resulting incidence angle):")
    targets = (15, 30, 60, 100, 150, 170, 200)
    hdr = "surface" + "".join(f"{t:>14d} m" for t in targets)
    print(f"{hdr}")
    step = dgrid[1] - dgrid[0]
    for kind in SURF:
        out = hits[kind]
        dh, xh, zh = out[..., 0], out[..., 1], out[..., 2]
        valid = np.isfinite(dh)
        jj = np.zeros(valid.shape, dtype=int)
        jj[valid] = np.clip(((dh[valid] - dgrid[0]) / step).astype(int),
                            0, len(dgrid) - 1)
        nvec = nrm[kind]                      # (n, 3)
        cosang = (np.abs(xh * nvec[:, 0][jj] - dh * nvec[:, 1][jj]
                         + zh * nvec[:, 2][jj])
                  / np.maximum(np.sqrt(xh ** 2 + dh ** 2 + zh ** 2),
                               1e-9))
        incs2 = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
        row = f"{NAMES[kind]:<9}"
        for t in targets:
            sel = valid & (dh > t - 3) & (dh <= t + 3)
            if not sel.any():
                row += f"{'  -':>14s}"
                continue
            m = np.where(sel, incs2, np.inf)
            k2 = np.unravel_index(m.argmin(), m.shape)
            row += (f"  {az[k2[0]]:>+5.0f}/{el[k2[1]]:>+4.0f} "
                    f"i{m[k2]:>3.0f}")
        print(row)

    # ---- actual far-range point incidence vs model -------------------
    print()
    print("[tunnel_model] incidence of ACTUAL far points (100-210 m):")
    r = open_reader(args.bag)
    f = 0
    acc = {k: [] for k in SURF}
    acc["other"] = []
    while r.has_next() and f < args.frames:
        _, d, _ = r.read_next()
        x, y, z, _ = parse_points(d)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[m], y[m], z[m]
        df = -y
        st = 5
        for i in range(0, len(x), st):
            dx_, dy_, dz_ = x[i], y[i], z[i]
            d_ = -dy_
            if not (100 <= d_ <= 210):
                continue
            j = int(np.clip((d_ - dgrid[0]) / (dgrid[1] - dgrid[0]),
                            0, len(dgrid) - 1))
            dirv = np.array([dx_, dy_, dz_])
            dl = np.linalg.norm(dirv)
            if dl < 1e-6:
                continue
            dirv /= dl
            def inc_of(nrm_):
                return np.degrees(np.arccos(np.clip(
                    abs(dirv @ nrm_[j]), 0, 1)))
            if abs(dx_ - model["x_left"][j]) < 0.6 and dx_ < 0:
                acc["x_left"].append(inc_of(nrm["x_left"]))
            elif abs(dx_ - model["x_right"][j]) < 0.6 and dx_ > 0:
                acc["x_right"].append(inc_of(nrm["x_right"]))
            elif abs(dz_ - model["z_floor"][j]) < 0.3:
                acc["z_floor"].append(inc_of(nrm["z_floor"]))
            elif abs(dz_ - model["z_ceil"][j]) < 0.3:
                acc["z_ceil"].append(inc_of(nrm["z_ceil"]))
            else:
                acc["other"].append(d_)
        f += 1
    for kind in list(SURF) + ["other"]:
        if kind == "other":
            if acc["other"]:
                a = np.array(acc["other"])
                print(f"  {'exit/other':<11} n={len(a):5d}  (not on any "
                      f"modelled surface)  d: med {np.median(a):.0f}, "
                      f"p25 {np.percentile(a,25):.0f}, p75 "
                      f"{np.percentile(a,75):.0f}")
            else:
                print(f"  {'exit/other':<11} n=0")
            continue
        if acc[kind]:
            a = np.array(acc[kind])
            print(f"  {NAMES[kind]:<11} n={len(a):5d}  median "
                  f"{np.median(a):5.1f} deg   p25 {np.percentile(a,25):4.1f}"
                  f"  p75 {np.percentile(a,75):4.1f}")
        else:
            print(f"  {NAMES[kind]:<11} n=0")

    # ---- far-end (170-200 m) wall positions --------------------------
    mfar = (dmid >= 170) & (dmid <= 200)
    if mfar.any():
        print()
        print("far-end wall positions, 170-200 m (median over the band):")
        print(f"  left wall:  x = {np.median(model['x_left'][mfar]):+.2f} m"
              f"   (range {model['x_left'][mfar].min():+.2f} .. "
              f"{model['x_left'][mfar].max():+.2f})")
        print(f"  right wall: x = {np.median(model['x_right'][mfar]):+.2f} m"
              f"   (range {model['x_right'][mfar].min():+.2f} .. "
              f"{model['x_right'][mfar].max():+.2f})")
        print(f"  floor:      z = {np.median(model['z_floor'][mfar]):+.2f} m"
              f"    ceiling: z = {np.median(model['z_ceil'][mfar]):+.2f} m")
        for name, curve, side in (("left wall", model["x_left"], "left"),
                                  ("right wall", model["x_right"],
                                   "right")):
            xm = float(np.median(curve[mfar]))
            if (side == "left") == (xm < 0):
                az_hit = np.degrees(np.arctan2(xm, 185.0))
                print(f"  beam to {name} at ~185 m:  azimuth "
                      f"{az_hit:+.1f} deg, elevation 0 (wall mid "
                      f"height); incidence ~"
                      f"{90 - abs(az_hit) - 0:.0f} deg (grazing - far "
                      f"walls are nearly perpendicular to the line of "
                      f"sight of a centred sensor)")
        zf = float(np.median(model["z_floor"][mfar]))
        zc = float(np.median(model["z_ceil"][mfar]))
        print(f"  beam to floor @185 m:  elevation "
              f"{-np.degrees(np.arctan2(-zf, 185.0)):+.1f} deg   "
              f"to ceiling: elevation "
              f"{np.degrees(np.arctan2(zc, 185.0)):+.1f} deg "
              f"(both grazing on flat surfaces)")

    # ---- plot (PIL; system matplotlib is not numpy2-safe) ----------
    try:
        def mkf(x0, x1, y0, y1, xr, yr):
            def f(px, py):
                fx = (px - xr[0]) / (xr[1] - xr[0])
                fy = (py - yr[0]) / (yr[1] - yr[0])
                return (x0 + fx * (x1 - x0), y1 - fy * (y1 - y0))
            return f

        img = Image.new("RGB", (1500, 480), "white")
        drw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        f0 = mkf(20, 740, 30, 330, (0, 215), (-8, 8))
        f1 = mkf(780, 1480, 30, 330, (0, 215), (-4, 7))
        drw.rectangle([20, 30, 740, 330], outline="black", width=1)
        drw.text((25, 35), f"{tag} TOP  (d vs x)", fill="black", font=font)
        drw.rectangle([780, 30, 1480, 330], outline="black", width=1)
        drw.text((785, 35), f"{tag} SIDE  (d vs z)", fill="black",
                 font=font)

        def curve(f, data, c, wd):
            drw.line([f(p, q) for p, q in zip(*data)], fill=c, width=wd)

        curve(f0, (dmid, model["x_left"]), (0, 0, 255), 2)
        curve(f0, (dmid, model["x_right"]), (255, 140, 0), 2)
        curve(f1, (dmid, model["z_floor"]), (0, 160, 0), 2)
        curve(f1, (dmid, model["z_ceil"]), (200, 0, 200), 2)
        curve(f1, (dmid, np.zeros_like(dmid)), (60, 60, 60), 1)

        for kind, c, ff, off in (("x_left", (255, 0, 0), f0, 0.0),
                                 ("x_right", (0, 120, 0), f0, 0.0),
                                 ("z_floor", (0, 160, 0), f1, 0.3),
                                 ("z_ceil", (200, 0, 200), f1, -0.6)):
            if kind in best:
                azb, elb, dh, xh, zh, inc = best[kind]
                drw.line([ff(0, 0), ff(dh, xh)], fill=c, width=1)
                drw.text(ff(dh * 0.4, zh if ff is f1 else xh + off + 0.3),
                         f"az={azb:+.0f} el={elb:+.0f} inc={inc:.0f}",
                         fill=c, font=font)

        valid = ((model["x_left"] < -0.5) & (model["x_right"] > 0.5)
                 & (model["z_ceil"] > 0.5) & (model["z_floor"] < 0.3))
        dvalid = dmid[valid]
        tl = float(dvalid.max()) if len(dvalid) else float("nan")
        drw.text((20, 350),
                 f"model-valid tunnel length (both walls + ceiling "
                 f"present): ~{tl:.0f} m; far range beyond that = "
                 f"outdoor/exit, not wall.  blue=left wall "
                 f"orange=right wall green=floor magenta=ceiling; thin "
                 f"lines = recommended steerable beams (az from "
                 f"forward, el up; inc = incidence, deg)",
                 fill="black", font=font)

        p = os.path.join(outdir, "tunnel_model.png")
        img.save(p)
        print(f"[tunnel_model] wrote {p}")
    except Exception as e:
        print(f"[tunnel_model] plot failed: {e}")

    print(f"[tunnel_model] done -> {outdir}")


if __name__ == "__main__":
    main()
