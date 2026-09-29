#!/usr/bin/env python3
"""
slice_detect.py - LiDAR obstacle detection for tunnel train movement.

Reads a rosbag2 (.db3) bag with the Hesai-128 PointCloud2 topic, cuts the
pointcloud into forward slices ("chunks") of N metres each (default 2 m)
ahead of the lidar, and for every chunk looks for obstacles inside a
detection rectangle 3 m wide by default.  By default the rectangle is
placed on the track: its lateral centre is exactly between the rails
(detected per frame from the point cloud; the lidar is mounted at the
centre of the train, 1.075 m above the rail head) and its bottom sits
on the rail level with the top 2.1 m above the rails -- i.e. anything
in this 3 x 2.1 m box is a dangerous obstacle.  An obstacle is any
point cluster whose bounding size exceeds MIN_SIZE (0.1 m) and has at
least MIN_PTS points.  Alternative centreing: --center tunnel
(per-chunk tunnel-axis midpoint) or --center 0 (fixed x=0).

Frame convention (verified on these bags): ego/lidar frame,
    forward = -y,  lateral = x,  up = +z.

Point layout (26 bytes, verified on raw CDR):
    x f32 @0, y f32 @4, z f32 @8, intensity f32 @12,
    ring u8 @16, timestamp u64 @18        (ring/ts ignored here)

Usage (ROS2 humble must be available for rosbag2_py):
    source /opt/ros/humble/setup.bash
    /usr/bin/python3.10 slice_detect.py for_hackathon/doubleT_obstacle
    /usr/bin/python3.10 slice_detect.py for_hackathon/doubleT_obstacle \
        --chunk 2 --min-size 0.1 --center rails --rail-z 1.075 \
        --rect-top 2.1 --stride 1 \
        --outdir out/doubleT_obstacle --stills 4

Outputs (per bag, into --outdir):
    detections.csv   one row per detected obstacle cluster
    summary.csv      per frame x chunk: rect point count, clusters, max size
    grid.png         frames (x) x chunks (y) heatmap of max cluster size
    still_*.png      annotated top-down (x vs forward) views of busy frames

Two passes over the bag: pass 1 does the detection, pass 2 re-reads only
to render the selected stills (keeps memory small).
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
except ImportError:
    sys.exit("ROS2 must be sourced first:  source /opt/ros/humble/setup.bash")

PT_STEP = 26
NEIGH13 = [(1, 0, 0), (-1, 1, 0), (0, 1, 0), (1, 1, 0),
           (1, -1, 0), (1, 0, 1), (-1, 1, 1), (0, 1, 1),
           (1, 1, 1), (1, -1, 1), (1, 0, -1), (-1, 1, -1), (0, 1, -1)]


# ---------------------------------------------------------------- parsing
def parse_points(data):
    """rosbag2 message data -> (x, y, z, intensity) arrays, junk removed."""
    msg = deserialize_message(data, PointCloud2)
    n = msg.height * msg.width
    d = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, PT_STEP)
    x = d[:, 0:4].copy().view(np.float32).ravel()
    y = d[:, 4:8].copy().view(np.float32).ravel()
    z = d[:, 8:12].copy().view(np.float32).ravel()
    i = d[:, 12:16].copy().view(np.float32).ravel()
    valid = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        & ~((x == 0) & (y == 0) & (z == 0))
        & (np.abs(x) < 100) & (np.abs(y) < 250) & (np.abs(z) < 50)
    )
    return x[valid], y[valid], z[valid], i[valid]


# ---------------------------------------------------------------- clustering
def grid_clusters(x, y, z, eps):
    """Union-find clustering on a 3D grid of cell size eps (~DBSCAN(eps,0)).

    x, y, z are 1-D float arrays of the same length.
    Returns (labels (n,), n_clusters).
    """
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.int32), 0
    eps2 = eps * eps
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cx = np.floor(x / eps).astype(np.int64)
    cy = np.floor(y / eps).astype(np.int64)
    cz = np.floor(z / eps).astype(np.int64)
    bycell = {}
    for k in range(n):
        bycell.setdefault((cx[k], cy[k], cz[k]), []).append(k)
    for (ix, iy, iz), a in bycell.items():
        a = np.fromiter(a, dtype=np.int64)
        for dx, dy, dz in NEIGH13:
            b = bycell.get((ix + dx, iy + dy, iz + dz))
            if b is None:
                continue
            b = np.fromiter(b, dtype=np.int64)
            # block-wise distance test (keeps memory bounded)
            for i0 in range(0, len(a), 64):
                aa = a[i0:i0 + 64]
                ax, ay, az = x[aa], y[aa], z[aa]
                for j0 in range(0, len(b), 64):
                    bb = b[j0:j0 + 64]
                    d2 = ((ax[:, None] - x[bb][None, :]) ** 2
                          + (ay[:, None] - y[bb][None, :]) ** 2
                          + (az[:, None] - z[bb][None, :]) ** 2)
                    hits = np.argwhere(d2 <= eps2)
                    if len(hits) == 0:
                        continue
                    if len(hits) > 256:
                        # dense match (continuous surface): merge all
                        # matched members of this cell pair in one go
                        ai = aa[hits[:, 0]]
                        bi = bb[hits[:, 1]]
                        anchor = ai[0]
                        for q in np.unique(np.concatenate([ai[1:], bi])):
                            union(q, anchor)
                    else:
                        for h in hits:
                            union(aa[h[0]], bb[h[1]])
    labels = np.array([find(k) for k in range(n)], dtype=np.int64)
    _, labels = np.unique(labels, return_inverse=True)
    return labels, len(labels) if n else 0


# ---------------------------------------------------------------- helpers
def _tcolor(t):
    t = np.clip(float(t), 0.0, 1.0)
    return (int(15 + 240 * t), int(20 + 60 * t), int(25 + 35 * (1 - t)))


def est_center_auto(x, d, z, rect_w, fallback=0.0):
    """Tunnel axis of a chunk = midpoint of the left/right wall extents
    (robust 1st/99th percentiles: interior masses like platforms do
    not move it).  Works while the tunnel curves or changes shape
    slowly; falls back to the previous estimate when walls are not
    visible (far field)."""
    if len(x) < 50:
        return fallback
    lo, hi = np.percentile(x, [1, 99])
    if hi - lo > max(4.0, rect_w):          # walls really visible
        return float((lo + hi) / 2)
    return fallback


def detect_rails(x, d, z, z_hint, gauge=1.52,
                 d_lo=2.0, d_hi=20.0, x_lo=-1.5, x_hi=1.5):
    """Find the rail top surfaces in one frame (template match).

    The lidar sits at the centre of the train, so the gauge centre is
    close to x=0: for candidate centre offsets c in [-0.35, +0.35] we
    count points in two narrow lateral bands at c +/- gauge/2 (the
    rail tops) within d in [2, 20] m and a z window around the last
    known rail level (wide on the first frame).  The best c wins; a
    side counts as a real rail only if its band has enough points
    with a flat top (z std <= 0.12 m).

    Returns (x_left, x_right, z_rail); a missing side is None, and
    z_rail is None if no real rail was found."""
    if z_hint is None:
        zlo, zhi = -2.2, -0.5                 # first frame: wide window
    else:
        zlo, zhi = z_hint - 0.30, z_hint + 0.18
    m = ((d >= d_lo) & (d <= d_hi)
         & (x >= x_lo) & (x <= x_hi) & (z >= zlo) & (z <= zhi))
    if not m.any():
        return None, None, None
    xs, zs = x[m], z[m]
    half = gauge / 2                          # 0.76
    band = 0.14                              # rail top +- 14 cm
    best_c, best_score = 0.0, -1.0
    for c in np.arange(-0.35, 0.351, 0.01):
        nL = int(((xs >= c - half - band) & (xs <= c - half + band)).sum())
        nR = int(((xs >= c + half - band) & (xs <= c + half + band)).sum())
        score = min(nL, nR) * 2 + (nL + nR) * 0.2   # reward balance
        if score > best_score:
            best_c, best_score = c, score
    c = best_c
    mL = (xs >= c - half - band) & (xs <= c - half + band)
    mR = (xs >= c + half - band) & (xs <= c + half + band)

    # per side: the rail top is the TOPMOST flat sub-band of the z
    # distribution in the band (in a wide z window the floor below
    # would otherwise win on point count)
    def side_stats(ms):
        n = int(ms.sum())
        if n < 25:
            return None, None
        szs, zx = zs[ms], xs[ms]
        if szs.std() <= 0.12:            # single flat level
            return float(zx.mean()), float(np.median(szs))
        best = (None, None)
        zb, zed = np.histogram(szs, bins=16,
                               range=[szs.min(), szs.max() + 1e-6])
        for i in range(len(zb) - 1):
            lo, hi = zed[i], zed[i + 1]
            if hi - lo > 0.14:
                continue
            sub = (szs >= lo) & (szs < hi + 1e-6)
            if sub.sum() < 25 or szs[sub].std() > 0.12:
                continue
            med = float(np.median(szs[sub]))
            if best[1] is None or med > best[1]:
                best = (float(zx[sub].mean()), med)
        return best

    xL, zL = side_stats(mL)
    xR, zR = side_stats(mR)
    if xL is None and xR is None:
        return None, None, None
    if xL is not None and xR is not None:
        return xL, xR, float(np.median([zL, zR]))
    if xR is not None:
        return xR - gauge, xR, zR          # mirror left from gauge
    return xL, xL + gauge, zL


def chunk_rect_points(x, d, z, inten, k, chunk, xc, half_w, z_lo, z_hi):
    m = ((d >= k * chunk) & (d < (k + 1) * chunk)
         & (np.abs(x - xc) <= half_w) & (z >= z_lo) & (z <= z_hi))
    return x[m], d[m], z[m], inten[m], m


def open_reader(bag):
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id="sqlite3"), ConverterOptions())
    return r


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="bag dir or .db3 path")
    ap.add_argument("--chunk", type=float, default=2.0,
                    help="slice width in metres ahead of the lidar (2)")
    ap.add_argument("--rect-w", type=float, default=3.0,
                    help="detection rectangle width (3)")
    ap.add_argument("--rect-h", type=float, default=2.1,
                    help="detection rectangle height for --center tunnel|0 "
                         "(2.1)")
    ap.add_argument("--z-center", type=float, default=0.0,
                    help="rectangle centre height in lidar frame for "
                         "--center tunnel|0 (0)")
    ap.add_argument("--center", default="rails",
                    choices=["0", "tunnel", "rails"],
                    help="lateral rectangle centre: 'rails' = exactly "
                         "between the rails (detected from the point "
                         "cloud, default), 'tunnel' = per-chunk wall "
                         "midpoint, '0' = fixed x=0")
    ap.add_argument("--rail-z", type=float, default=1.075,
                    help="lidar height above rail head in m (1.075); "
                         "initial rail-level estimate and fallback")
    ap.add_argument("--rect-top", type=float, default=2.1,
                    help="--center rails: rectangle top height above "
                         "rail head in m (2.1); bottom sits on the rails")
    ap.add_argument("--gauge", type=float, default=1.52,
                    help="track gauge in m (1.52); used when only one "
                         "rail is visible")
    ap.add_argument("--min-size", type=float, default=0.1,
                    help="obstacle size threshold in m (0.1)")
    ap.add_argument("--min-pts", type=int, default=3,
                    help="minimum cluster size in points (3)")
    ap.add_argument("--dmin", type=float, default=0.5,
                    help="ignore points closer than this (sensor mount)")
    ap.add_argument("--dmax", type=float, default=60.0,
                    help="max forward detection range in m (60)")
    ap.add_argument("--eps", type=float, default=0.25,
                    help="clustering neighbourhood in m (0.25)")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (1)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = all)")
    ap.add_argument("--outdir", default="",
                    help="output dir (default out/<bagname>)")
    ap.add_argument("--stills", type=int, default=4,
                    help="number of annotated stills to save (4)")
    args = ap.parse_args()

    if args.bag.endswith(".db3"):
        bag = args.bag
    elif os.path.isdir(args.bag):
        dbs = sorted(glob.glob(os.path.join(args.bag, "*.db3")))
        if not dbs:
            sys.exit(f"no .db3 found under {args.bag}")
        bag = dbs[0]
    else:
        bag = args.bag
    tag = os.path.basename(os.path.dirname(bag)) or \
        os.path.splitext(os.path.basename(bag))[0]
    outdir = args.outdir or os.path.join("out", tag)
    os.makedirs(outdir, exist_ok=True)

    half_w = args.rect_w / 2
    if args.center == "rails":
        # rectangle sits on the rails: bottom = rail level, top =
        # rail level + rect-top; both tracked per frame
        z_lo = -args.rail_z - 0.30     # initial (refined by detection)
        z_hi = z_lo + args.rect_top
    else:
        z_lo = args.z_center - args.rect_h / 2
        z_hi = args.z_center + args.rect_h / 2
    kmax = int(np.ceil(args.dmax / args.chunk))

    # ============================ PASS 1: detection =====================
    t0 = time.time()
    nframes = 0
    detections = []
    summary = []
    chunk_max = {}          # (frame, chunk) -> max cluster size
    chist = {}              # frame -> {chunk: raw centre estimate} (tunnel)
    railhist = {}           # frame -> (xc_raw, zr_raw) (rails)
    still_want = {}         # frame -> (chunk, max_size)
    xc_of = {}             # (frame, chunk) -> smoothed centre (for pass2)
    zr_of = {}             # frame -> smoothed rail level (for pass2)

    r = open_reader(bag)
    while r.has_next():
        _, data, ts = r.read_next()
        if nframes % args.stride != 0:
            nframes += 1
            continue
        if args.max_frames and nframes >= args.max_frames:
            break
        nframes += 1

        x, y, z, inten = parse_points(data)
        d = -y                                   # forward distance
        keep = (d >= args.dmin) & (d <= args.dmax)
        x, d, z, inten = x[keep], d[keep], z[keep], inten[keep]
        if len(x) == 0:
            continue

        # ---------------- rail-based centre (once per frame) --------
        if args.center == "rails":
            # last real detection in the recent past -> z window hint
            hint = None
            for f in range(nframes - 1, max(0, nframes - 8), -1):
                if f in railhist and railhist[f][2]:
                    hint = railhist[f][1]
                    break
            # (hint stays None until a real detection exists -> wide
            #  window, which also finds rails far from the spec level)
            xL, xR, zr = detect_rails(x, d, z, hint, gauge=args.gauge)
            found = zr is not None
            if found:
                raw_xc = (xL + xR) / 2
                # rail level cannot jump more than 25 cm between
                # nearby frames (track gradient); reject wild values
                for f in range(nframes - 1, max(0, nframes - 8), -1):
                    if f in railhist and railhist[f][2]:
                        if abs(zr - railhist[f][1]) > 0.25:
                            zr = railhist[f][1]
                        break
            else:                            # spec fallback
                raw_xc, zr = 0.0, -args.rail_z
            railhist[nframes] = (raw_xc, zr, found)
            real = [railhist[f] for f in range(nframes - 6,
                                               nframes + 1)
                    if f in railhist and railhist[f][2]]
            xc = float(np.median([v[0] for v in real])) if real \
                else raw_xc
            zr = float(np.median([v[1] for v in real])) if real \
                else (-args.rail_z)
            if len(railhist) > 20:
                for old in [f for f in railhist if f < nframes - 20]:
                    del railhist[old]
            zr_of[nframes] = zr
            z_lo_f, z_hi_f = zr, zr + args.rect_top
        else:
            zr_of[nframes] = None
            z_lo_f, z_hi_f = z_lo, z_hi

        # temporal smoothing of the per-chunk centre estimates: keep a
        # short history and take the median (slow curves, stable output)
        chist[nframes] = {}
        best_this_frame = (0.0, 0)
        for kk in range(kmax):
            m_chunk = (d >= kk * args.chunk) & (d < (kk + 1) * args.chunk)
            if not m_chunk.any():
                continue
            if args.center == "0":
                raw_c = 0.0
            elif args.center == "rails":
                raw_c = xc                     # same for every chunk
            else:
                # fallback chain: previous chunk of this frame, then
                # same chunk of the previous frame, then 0
                fb = 0.0
                if kk > 0 and kk - 1 in chist[nframes]:
                    fb = chist[nframes][kk - 1]
                elif nframes > 1 and kk in chist.get(nframes - 1, {}):
                    fb = chist[nframes - 1][kk]
                raw_c = est_center_auto(x[m_chunk], d[m_chunk], z[m_chunk],
                                        args.rect_w, fallback=fb)
            chist[nframes][kk] = raw_c
            if args.center == "rails":
                xc = raw_c
            else:
                # median over last 11 frames of this chunk
                recent = [chist[f][kk] for f in range(nframes - 10,
                                                      nframes + 1)
                          if f in chist and kk in chist[f]]
                xc = float(np.median(recent)) if recent else raw_c
            xc_of[(nframes, kk)] = xc
            if args.center != "rails" and len(chist) > 15:
                oldest = min(chist)
                if oldest < nframes - 15:
                    del chist[oldest]
            xr, dr, zr_, ir, _ = chunk_rect_points(
                x, d, z, inten, kk, args.chunk, xc, half_w, z_lo_f, z_hi_f)
            if len(xr) < args.min_pts:
                continue
            labels, nc = grid_clusters(xr, dr, zr_, args.eps)
            max_size, max_npts, max_cent = 0.0, 0, None
            for c in range(nc):
                cm = labels == c
                npts = int(cm.sum())
                if npts < args.min_pts:
                    continue
                sx = xr[cm].max() - xr[cm].min()
                sd = dr[cm].max() - dr[cm].min()
                sz = zr_[cm].max() - zr_[cm].min()
                size = max(sx, sd, sz)
                if size < args.min_size:
                    continue                    # noise speck
                detections.append({
                    "frame": nframes, "t_ns": ts, "chunk": kk,
                    "d_range": f"{kk*args.chunk:.1f}-{(kk+1)*args.chunk:.1f}",
                    "center_x": f"{xc:.2f}",
                    "cx": f"{xr[cm].mean():.2f}",
                    "d_fwd": f"{dr[cm].mean():.2f}",
                    "cz": f"{zr_[cm].mean():.2f}",
                    "size_x": f"{sx:.2f}", "size_d": f"{sd:.2f}",
                    "size_z": f"{sz:.2f}", "max_size": f"{size:.2f}",
                    "npts": npts, "i_mean": f"{ir[cm].mean():.0f}",
                })
                if size > max_size:
                    max_size, max_npts, max_cent = size, npts, (
                        float(xr[cm].mean()), float(dr[cm].mean()),
                        float(zr_[cm].mean()), npts)
            summary.append([nframes, ts, kk,
                            f"{kk*args.chunk:.1f}-{(kk+1)*args.chunk:.1f}",
                            f"{xc:.2f}", len(xr), nc,
                            f"{max_size:.2f}", max_npts])
            chunk_max[(nframes, kk)] = max_size
            if max_size > best_this_frame[0]:
                best_this_frame = (max_size, kk)
        if best_this_frame[0] >= args.min_size:
            prev = still_want.get(nframes)
            if prev is None or best_this_frame[0] > prev[1]:
                still_want[nframes] = (best_this_frame[1],
                                       best_this_frame[0])
        if nframes % 25 == 0:
            print(f"  [{tag}] pass1 frame {nframes}  "
                  f"{(time.time() - t0):.0f}s", flush=True)
    print(f"[{tag}] pass1: {nframes} frames in {time.time() - t0:.1f}s, "
          f"{len(detections)} detections")

    # ---------------------------------------------------------- outputs
    det_path = os.path.join(outdir, "detections.csv")
    if detections:
        with open(det_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detections[0].keys()))
            w.writeheader()
            w.writerows(detections)
    else:
        open(det_path, "w").close()
    sum_path = os.path.join(outdir, "summary.csv")
    with open(sum_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_ns", "chunk", "d_range", "center_x",
                    "npts_in_rect", "n_clusters", "max_size", "max_npts"])
        w.writerows(summary)
    print(f"[{tag}] wrote {det_path} ({len(detections)} rows), "
          f"{sum_path} ({len(summary)} rows)")

    # ---- stable objects: consolidate repeated detections over time.
    # In a static (or slowly moving) scene the same physical object is
    # detected in many consecutive frames; group detections by (chunk,
    # centroid proximity) into one row per object.
    objs = []                       # list of dicts, samples = list of
                                    # (frame, cx, cz, dfwd, size, npts)
    for det in detections:
        kk = det["chunk"]
        cx, cz = float(det["cx"]), float(det["cz"])
        df = float(det["d_fwd"])
        best = None
        for o in objs:
            if o["chunk"] != kk:
                continue
            if (abs(o["cx"] - cx) < 0.6 and abs(o["cz"] - cz) < 0.6
                    and abs(o["d_fwd"] - df) < 1.0):
                best = o
                break
        if best is None:
            best = {
                "chunk": kk, "d_range": det["d_range"],
                "samples": [], "center_x": det["center_x"],
                "cx": cx, "cz": cz, "d_fwd": df, "nd": 0,
            }
            objs.append(best)
        best["samples"].append((det["frame"], cx, cz, df,
                                float(det["max_size"]), int(det["npts"])))
        # running centroid for the matching test
        best["nd"] += 1
        best["cx"] = (best["cx"] * (best["nd"] - 1) + cx) / best["nd"]
        best["cz"] = (best["cz"] * (best["nd"] - 1) + cz) / best["nd"]
        best["d_fwd"] = (best["d_fwd"] * (best["nd"] - 1) + df) / best["nd"]
    nframes_total = nframes
    obj_rows = []
    for o in objs:
        # one sample per frame (first seen), then average
        byframe = {}
        for s in o["samples"]:
            byframe.setdefault(s[0], s)
        ss = list(byframe.values())
        n = len(ss)
        fr = sorted(s[0] for s in ss)
        cx = float(np.mean([s[1] for s in ss]))
        cz = float(np.mean([s[2] for s in ss]))
        df = float(np.mean([s[3] for s in ss]))
        size = max(s[4] for s in ss)
        npts = max(s[5] for s in ss)
        obj_rows.append({
            "chunk": o["chunk"], "d_range": o["d_range"],
            "seen_in_frames": n,
            "persistence": f"{n / max(1, nframes_total):.2f}",
            "frame_range": f"{fr[0]}-{fr[-1]}",
            "center_x": o["center_x"], "cx": f"{cx:.2f}",
            "d_fwd": f"{df:.2f}", "cz": f"{cz:.2f}",
            "max_size": f"{size:.2f}", "npts": npts,
            "kind": ("STABLE" if n >= max(3, 0.3 * nframes_total)
                    else "transient"),
        })
    obj_rows.sort(key=lambda r: (-int(r["seen_in_frames"]), r["chunk"]))
    obj_path = os.path.join(outdir, "objects.csv")
    if obj_rows:
        with open(obj_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(obj_rows[0].keys()))
            w.writeheader()
            w.writerows(obj_rows)
    else:
        open(obj_path, "w").close()
    n_stable = sum(1 for r in obj_rows if r["kind"] == "STABLE")
    print(f"[{tag}] wrote {obj_path} ({len(obj_rows)} consolidated "
          f"objects, {n_stable} stable)")

    # grid heat-map: rows = chunks, cols = frames
    if chunk_max:
        fmin = min(f for f, _ in chunk_max)
        fmax = max(f for f, _ in chunk_max)
        ktop = max(k for _, k in chunk_max)
        vals = np.full((ktop + 1, fmax - fmin + 1), np.nan)
        for (f, kk), v in chunk_max.items():
            vals[kk, f - fmin] = v
        with np.errstate(invalid="ignore"):
            norm = np.where(vals > 0,
                            np.log10(1 + vals) / np.log10(1 + 2.0), 0)
        cw, chh = 6, 18
        img = Image.new("RGB",
                        (60 + (fmax - fmin + 1) * cw + 40,
                         40 + (ktop + 1) * chh + 30), (12, 14, 16))
        drw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=12)
        except (AttributeError, TypeError):
            font = ImageFont.load_default()
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                col = (20, 22, 25) if np.isnan(v) else _tcolor(norm[i, j])
                drw.rectangle([60 + j * cw, 40 + i * chh,
                               60 + j * cw + cw - 1, 40 + i * chh + chh - 1],
                              fill=col)
        drw.text((8, 10),
                 f"{tag}: max cluster size per chunk/frame (log, red=big)",
                 fill=(230, 230, 230), font=font)
        for kk in range(0, ktop + 1, 2):
            drw.text((2, 40 + kk * chh + 3), f"{kk*args.chunk:.0f}m",
                     fill=(160, 160, 160), font=font)
        drw.text((60, 40 + (ktop + 1) * chh + 8),
                 f"frame ->   ({fmin} .. {fmax})",
                 fill=(160, 160, 160), font=font)
        grid_path = os.path.join(outdir, "grid.png")
        img.save(grid_path)
        print(f"[{tag}] wrote {grid_path}")

    if not still_want or args.stills == 0:
        print(f"[{tag}] done (no stills)")
        return

    # ========================= PASS 2: annotated stills ================
    picked = sorted(still_want.items(), key=lambda kv: -kv[1][1])[:args.stills]
    want = {f: (kk, sz) for f, (kk, sz) in picked}
    print(f"[{tag}] pass2: rendering stills for frames "
          f"{[f for f, _ in picked]}")
    t1 = time.time()
    got = set()
    r = open_reader(bag)
    nframes = 0
    while r.has_next():
        _, data, ts = r.read_next()
        if nframes % args.stride != 0:
            nframes += 1
            continue
        if nframes in want and nframes not in got:
            kk, sz = want[nframes]
            x, y, z, inten = parse_points(data)
            d = -y
            keep = (d >= args.dmin) & (d <= args.dmax)
            x, d, z, inten = x[keep], d[keep], z[keep], inten[keep]
            m_chunk = (d >= kk * args.chunk) & (d < (kk + 1) * args.chunk)
            xc = xc_of.get((nframes, kk), 0.0)   # exact pass1 centre
            zr2 = zr_of.get(nframes)
            if args.center == "rails" and zr2 is not None:
                z_lo2, z_hi2 = zr2, zr2 + args.rect_top
            else:
                z_lo2, z_hi2 = z_lo, z_hi
            xr, dr, zr, ir, _ = chunk_rect_points(
                x, d, z, inten, kk, args.chunk, xc, half_w, z_lo2, z_hi2)
            if len(xr) < args.min_pts:
                xr, dr, zr, ir = xr[:0], dr[:0], zr[:0], ir[:0]
            render_still(outdir, tag, nframes, kk, xc, x, d, z,
                         xr, dr, zr, ir, args, sz, zr2)
            got.add(nframes)
        nframes += 1
        if args.max_frames and nframes >= args.max_frames:
            break
        if len(got) == len(want):
            break
    print(f"[{tag}] pass2: {len(got)} stills in {time.time() - t1:.1f}s "
          f"-> {outdir}")


# ---------------------------------------------------------------- stills
def render_still(outdir, tag, nframes, kk, xc, x, d, z, xr, dr, zr, ir,
                 args, chunk_size, zr2=None):
    """Top-down view (x vs forward) with chunk grid, rect and obstacles."""
    W, H = 1600, 700
    img = Image.new("RGB", (W, H), (12, 14, 16))
    drw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=12)
    except (AttributeError, TypeError):
        font = ImageFont.load_default()
    xmin, xmax = -9, 9
    dmaxv = min(60.0, args.dmax)
    dminv = 0.0
    px = lambda xx: int(80 + (xx - xmin) / (xmax - xmin) * (W - 160))
    py = lambda dd: int(H - 60 - (dd - dminv) / (dmaxv - dminv) * (H - 120))

    # chunk grid lines
    for cc in range(0, int(dmaxv // args.chunk) + 1):
        xx = px(cc * args.chunk)
        drw.line([(xx, 60), (xx, H - 60)], fill=(30, 34, 40))
        drw.text((xx + 2, H - 56), f"{cc*args.chunk}m",
                 fill=(120, 120, 120), font=font)

    # detection rectangle of the selected chunk
    drw.rectangle([px(xc - args.rect_w / 2), py((kk + 1) * args.chunk),
                   px(xc + args.rect_w / 2), py(kk * args.chunk)],
                  outline=(90, 200, 120), width=2)

    # all points (subsampled), dim blue
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(x), size=min(50000, len(x)), replace=False)
    for ix in idxs:
        if xmin <= x[ix] <= xmax:
            drw.point((px(x[ix]), py(d[ix])), fill=(60, 80, 110))

    # rectangle points, cyan
    if len(xr):
        jr = rng.choice(len(xr), size=min(40000, len(xr)), replace=False)
        for ix in jr:
            drw.point((px(xr[ix]), py(dr[ix])), fill=(80, 220, 220))

    # detected clusters: red bounding boxes
    if len(xr):
        labels, nc = grid_clusters(xr, dr, zr, args.eps)
        for c in range(nc):
            cm = labels == c
            npts = int(cm.sum())
            if npts < args.min_pts:
                continue
            sx = xr[cm].max() - xr[cm].min()
            sd = dr[cm].max() - dr[cm].min()
            sz = zr[cm].max() - zr[cm].min()
            if max(sx, sd, sz) < args.min_size:
                continue
            drw.rectangle(
                [px(xr[cm].min()), py(dr[cm].max()),
                 px(xr[cm].max()), py(dr[cm].min())],
                outline=(255, 80, 60), width=2)
            drw.text((px(xr[cm].max()) + 3, py(dr[cm].max())),
                     f"{max(sx, sd, sz):.2f}m/{npts}p",
                     fill=(255, 120, 100), font=font)

    h_label = (args.rect_top if args.center == "rails" else args.rect_h)
    drw.text((80, 8),
             f"{tag}  frame {nframes}  chunk {kk} "
             f"({kk*args.chunk:.0f}-{(kk+1)*args.chunk:.0f} m)  rect "
             f"{args.rect_w}x{h_label} m, centre x={xc:.2f}" +
             (f", rail z={zr2:.2f}" if zr2 is not None else "") +
             "  |  "
             f"top-down, forward to the right  |  chunk max {chunk_size:.2f} m",
             fill=(230, 230, 230), font=font)
    path = os.path.join(outdir, f"still_f{nframes:03d}_k{kk:02d}.png")
    img.save(path)
    return path


if __name__ == "__main__":
    main()
