#!/usr/bin/env python3
"""
measure.py -- runtime distance measurement for the 3-laser + camera sensor.

Usage:
    python3 measure.py IMAGE.jpg [--cal calibration.json] [--out annotated.jpg]
    python3 measure.py --video VID.mp4 --cal calibration.json
    python3 measure.py --cam 0 --cal calibration.json

Prints for the current frame:
  * per-laser distance (left V1 / right V2 / bottom B),
  * the closest-object distance,
  * the two boundary corner points (V1 x B and V2 x B) in 2D image
    coordinates and in 3D camera coordinates (x right, y up, z forward).
Objects must stay outside the corridor between the two corner lines.

With --video/--cam it processes frames in a loop (q to quit).
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laser_sensor as ls


COLORS = {"V1": (0, 255, 0), "V2": (0, 160, 255), "H": (255, 0, 0)}
KEY_OF = {"L": "V1", "R": "V2", "B": "H"}


def segment_from_line(line: ls.ImageLine, gray: np.ndarray, margin: float = 4.0,
                      min_px: int = 30) -> ls.LineSegment:
    """Build a LineSegment around a fitted line from the bright pixels near
    it (used for the guided fallback when plain detection misses a line)."""
    H, W = gray.shape
    ys, xs = np.mgrid[0:H, 0:W]
    d = np.abs(line.a * xs + line.b * ys + line.c)
    m = d < margin
    # restrict to the central window (laser lines always live there)
    if line.is_vertical:
        m &= (xs > 0.25 * W) & (xs < 0.75 * W)
    else:
        m &= (ys > 0.25 * H) & (ys < 0.75 * H)
    n = int(m.sum())
    if n < min_px:
        return ls.LineSegment(line=line, t_start=-1e9, t_stop=1e9,
                              brightness=0.0, n_points=n)
    # t along the line direction (b, -a) from the origin
    dd = np.array([line.b, -line.a])
    t = xs[m] * dd[0] + ys[m] * dd[1]
    return ls.LineSegment(line=line, t_start=float(t.min()), t_stop=float(t.max()),
                          brightness=float(gray[m].mean()), n_points=n)


def measure_frame(model: ls.Model, maps: dict, img: np.ndarray,
                  min_brightness: float = None,
                  tracker: "NearTracker" = None) -> ls.Measurement:
    """Run the measurement; if a laser line is still missing, retry with
    guided extraction on a grid of distance guesses."""
    res = ls.measure(model, maps, img, min_brightness=min_brightness)
    missing = [fn for fn in ("L", "R", "B") if fn not in res.distances]
    if not missing or not maps:
        _annotate_near(res, img, maps, model, min_brightness=min_brightness,
                       tracker=tracker)
        return res
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    segs = {k: list(v) for k, v in getattr(res, "_segs", {}).items()}
    ctx = ls.guided_context(img)
    # distance guesses: first around the distances of the lasers that were
    # detected (lines usually share similar distance), then a coarse grid
    ds_try = []
    for d0 in list(res.distances.values()):
        for off in (0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0):
            d = d0 + off
            if 1.8 <= d <= 14.0:
                ds_try.append(d)
    ds_try += list(np.arange(2.0, 14.01, 1.0))
    seen = set()
    for fn in missing:
        key = KEY_OF[fn]
        best = None
        for d in ds_try:
            if abs(d) in seen:
                continue
            seen.add(abs(d))
            line, err = ls.extract_line_guided(img, model, fn, float(d),
                                               strip=50, max_err=15.0,
                                               ctx=ctx)
            if line.ok:
                best = (line, err)
                break
        if best is not None:
            segs[key] = [segment_from_line(best[0], g)]
    res = ls.measure(model, maps, img, min_brightness=min_brightness,
                     segs=segs)
    _annotate_near(res, img, maps, model, min_brightness=min_brightness,
                   tracker=tracker)
    return res


class NearTracker:
    """Temporal persistence filter for near-field features.

    A feature is reported only if it (re)appears in every one of the last
    `persist` measured frames (matched by overlapping bounding boxes), so
    single-frame speckle blips are never shown.
    """

    def __init__(self, persist: int = 3, dilate: float = 10.0):
        self.persist = persist
        self.dilate = dilate
        self.hist = []          # list of feature lists, oldest first

    @staticmethod
    def _overlap(f1, f2, dilate) -> bool:
        return not (f1["x1"] + dilate < f2["x0"] - dilate or
                    f2["x1"] + dilate < f1["x0"] - dilate or
                    f1["y1"] + dilate < f2["y0"] - dilate or
                    f2["y1"] + dilate < f1["y0"] - dilate)

    def _matched(self, f, past) -> bool:
        return any(self._overlap(f, p, self.dilate) for p in past)

    def update(self, features: list) -> list:
        self.hist.append(features)
        if len(self.hist) > self.persist:
            self.hist.pop(0)
        if len(self.hist) < self.persist:
            return []                       # window still filling
        past_frames = self.hist[:-1]
        return [f for f in features
                if all(self._matched(f, pf) for pf in past_frames)]


def _annotate_near(res: ls.Measurement, img: np.ndarray, maps: dict,
                   model, min_brightness=None, tracker=None) -> None:
    """Run the corridor near-feature detector on the final measurement and
    fold its (possibly smaller) distance into res.closest."""
    if not getattr(res, "_segs", None):
        return
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    thr = ls.image_threshold(g)
    if min_brightness is not None:
        thr = max(thr, float(min_brightness))
    # measured B segments with their distances (matched by midpoint/n)
    h_segs = []
    detB = res.details.get("B", [])
    for s in res._segs.get("H", []):
        mp = s.midpoint()
        for dd in detB:
            if dd["n"] == s.n_points and \
                    abs(dd["midpoint"][0] - mp[0]) < 1.5 and \
                    abs(dd["midpoint"][1] - mp[1]) < 1.5:
                h_segs.append((s.line, dd["d"]))
                break
    b_map = maps.get("B") if maps else None
    feats = ls.find_near_features(g, thr, res._segs,
                                  b_map, h_segs=h_segs,
                                  model=model,
                                  laser_ds=list(res.distances.values()))
    if tracker is not None:
        feats = tracker.update(feats)   # keep only 3-frame-persistent ones
    res.near_features = feats
    nd = [f["d"] for f in res.near_features if f["near"] and f["d"] is not None]
    res.near_min = min(nd) if nd else None
    if res.near_min is not None and (res.closest is None or res.near_min < res.closest):
        res.closest = res.near_min


def _line_endpoints_in_rect(a, b, c, W, H):
    """Clip the infinite line a*x + b*y + c = 0 to the image rectangle.
    Returns the two extreme endpoints (works even if the line only
    grazes the rectangle)."""
    cands = []
    if abs(b) > 1e-12:
        for x in (0.0, W - 1.0):
            y = -(a * x + c) / b
            if -1.0 <= y <= H:
                cands.append((x, y))
    if abs(a) > 1e-12:
        for y in (0.0, H - 1.0):
            x = -(b * y + c) / a
            if -1.0 <= x <= W:
                cands.append((x, y))
    if len(cands) < 2:
        corners = [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]
        corners.sort(key=lambda p: abs(a * p[0] + b * p[1] + c))
        return (int(corners[0][0]), int(corners[0][1])), \
               (int(corners[1][0]), int(corners[1][1]))
    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            d2 = (cands[i][0] - cands[j][0]) ** 2 + (cands[i][1] - cands[j][1]) ** 2
            if best is None or d2 > best[0]:
                best = (d2, cands[i], cands[j])
    return (int(best[1][0]), int(best[1][1])), (int(best[2][0]), int(best[2][1]))


TRAJ_COLORS = {"LxB": (255, 255, 0), "RxB": (255, 0, 255)}


def annotate(img: np.ndarray, res: ls.Measurement,
             traj: dict = None) -> np.ndarray:
    out = img.copy()
    # segments
    segs = getattr(res, "_segs", None) or {}
    for key in ("H", "V1", "V2"):
        for s in segs.get(key, []):
            l = s.line
            dd = np.array([l.b, -l.a])
            base = np.array([-l.a * l.c, -l.b * l.c])
            p1 = base + s.t_start * dd
            p2 = base + s.t_stop * dd
            cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                     COLORS[key], 2)
    # corners
    for c, c3 in zip(res.corners or (), res.corners_3d or ()):
        cv2.circle(out, (int(c[0]), int(c[1])), 6, (255, 0, 255), 2)
        cv2.circle(out, (int(c[0]), int(c[1])), 2, (255, 0, 255), -1)
    H, W = img.shape[:2]
    # --- vanishing points + corner trajectories (d -> infinity) ---
    if traj:
        for key, ci in (("LxB", 0), ("RxB", 1)):
            if key not in traj:
                continue
            col = TRAJ_COLORS[key]
            a, b, c = traj[key]["line"]
            vx, vy = traj[key]["vp"]
            p1, p2 = _line_endpoints_in_rect(a, b, c, W, H)
            cv2.line(out, p1, p2, col, 1, cv2.LINE_AA)
            vxi, vyi = int(vx), int(vy)
            if 0 <= vxi <= W - 1 and 0 <= vyi <= H - 1:
                cv2.circle(out, (vxi, vyi), 8, col, 1, cv2.LINE_AA)
                cv2.circle(out, (vxi, vyi), 2, col, -1, cv2.LINE_AA)
                cv2.line(out, (vxi - 13, vyi), (vxi + 13, vyi), col, 1, cv2.LINE_AA)
                cv2.line(out, (vxi, vyi - 13), (vxi, vyi + 13), col, 1, cv2.LINE_AA)
                cv2.putText(out, key + " @ d=inf", (vxi + 16, vyi - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            # line connecting the vanishing point with the current corner
            if ci < len(res.corners or ()):
                cxp, cyp = res.corners[ci]
                cv2.line(out, (vxi, vyi), (int(cxp), int(cyp)), col, 2, cv2.LINE_AA)
    # --- big min-distance stamp (top center) ---
    cl = res.closest
    if cl is None:
        big, col = "MIN --  no laser detected", (200, 200, 200)
    elif cl < 1.5:
        big, col = f"MIN {cl:.2f} m", (0, 0, 255)
    elif cl < 3.0:
        big, col = f"MIN {cl:.2f} m", (0, 200, 255)
    else:
        big, col = f"MIN {cl:.2f} m", (0, 220, 0)
    org = (W // 2, 46)
    (tw, th), baseline = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
    cv2.rectangle(out, (org[0] - tw // 2 - 10, org[1] - th - 8),
                  (org[0] + tw // 2 + 10, org[1] + baseline + 6), (0, 0, 0), -1)
    cv2.putText(out, big, (org[0] - tw // 2, org[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, "min distance to obstacle", (org[0] - tw // 2, org[1] + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    # --- per-laser text (top left) ---
    d = res.distances
    txt = "L=%s  R=%s  B=%s" % (
        (f"{d['L']:.2f}" if "L" in d else "--"),
        (f"{d['R']:.2f}" if "R" in d else "--"),
        (f"{d['B']:.2f}" if "B" in d else "--"))
    cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2)
    if res.corners and res.corners_3d:
        for i, c3 in enumerate(res.corners_3d):
            x3, y3, z3 = c3
            cv2.putText(out, f"c{i + 1} ({x3:.2f},{y3:.2f},{z3:.2f})",
                        (10, 60 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
    # --- near-obstacle candidates: bounding box around each spot ---
    for f in (res.near_features or ()):
        col = (0, 255, 255) if f.get("near") else (255, 255, 255)  # yellow/white
        pad = 4
        cv2.rectangle(out, (f["x0"] - pad, f["y0"] - pad),
                      (f["x1"] + pad, f["y1"] + pad), col, 1)
        if f.get("near"):
            label = (f"NEAR {f['d']:.2f} m" if f["d"] is not None else "NEAR?")
            cv2.putText(out, label, (f["x1"] + 8, max(12, f["y0"] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out


def print_result(res: ls.Measurement):
    print()
    print("per-laser distance (closest segment of each laser):")
    for fn, key in (("L", "V1"), ("R", "V2"), ("B", "H")):
        if fn in res.distances:
            segs = res.details.get(fn, [])
            seg = min(segs, key=lambda s: s["d"]) if segs else None
            if seg:
                print(f"  {fn} ({key:2s}): d={seg['d']:7.3f} m   "
                      f"(err {seg['err_px']:5.2f} px, quality {seg['quality']:6.2f})")
            else:
                print(f"  {fn} ({key:2s}): d={res.distances[fn]:7.3f} m")
        else:
            print(f"  {fn} ({key:2s}): not detected")
    cl = res.closest
    print(f"\nCLOSEST OBJECT DISTANCE: "
          f"{('%.3f m' % cl) if cl is not None else 'not detected'}")
    if res.near_features:
        print("\ncorridor features (bright spots not on the laser lines):")
        for f in res.near_features:
            dstr = f"{f['d']:.3f} m" if f["d"] is not None else "      ?"
            tag = ("NEAR: closer than main surface" if f.get("near")
                   else "other")
            print(f"  {f['kind']:5s} box=({f['x0']:4d},{f['y0']:3d})-"
                  f"({f['x1']:4d},{f['y1']:3d})  {f['region']:6s}  "
                  f"d={dstr}  [{tag}]")
        if res.near_min is not None:
            print(f"  -> near-field min: {res.near_min:.3f} m")
    if res.corners and res.corners_3d:
        names = ("left  (V1 x B)", "right (V2 x B)")
        print("\nboundary corners (keep objects out of the corridor between):")
        for i, (c, c3) in enumerate(zip(res.corners, res.corners_3d)):
            (x1, y1) = c
            (x13, y13, z13) = c3
            print(f"  {names[i % 2]}: img=({x1:6.1f},{y1:6.1f})   "
                  f"3D=({x13:6.3f},{y13:6.3f},{z13:6.3f}) m   d={z13:.3f} m")
    else:
        print("\nboundary corners: not detected")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="input image (BGR)")
    ap.add_argument("--video", help="process a video file instead")
    ap.add_argument("--cam", type=int, help="process camera index instead")
    ap.add_argument("--cal", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "calibration.json"))
    ap.add_argument("--out", help="where to save the annotated image (single image mode)")
    ap.add_argument("--save-video", help="output annotated video file (video/camera mode)")
    ap.add_argument("--min-brightness", type=float, default=None,
                    help="ignore segments dimmer than this (0..255)")
    args = ap.parse_args()

    if not (args.image or args.video or args.cam is not None):
        ap.error("provide an image, --video or --cam")

    model, doc, maps = ls.load_calibration(args.cal)
    if not maps:
        print("WARNING: calibration has no 1-D maps; using the 3-D model only")
    # corner trajectories: the path each boundary corner slides along as the
    # distance grows, plus its limit (vanishing) point at d -> infinity
    traj = ls.corner_trajectories(maps) if maps else None
    if traj:
        for k, v in traj.items():
            print(f"vanishing point {k}: ({v['vp'][0]:.0f}, {v['vp'][1]:.0f})")

    src = None
    if args.cam is not None:
        src = cv2.VideoCapture(args.cam)
        if not src.isOpened():
            raise SystemExit(f"cannot open camera {args.cam}")
    elif args.video:
        src = cv2.VideoCapture(args.video)
        if not src.isOpened():
            raise SystemExit(f"cannot open video {args.video}")

    writer = None
    if args.save_video and src is not None:
        w = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = src.get(cv2.CAP_PROP_FPS) or 15.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (w, h))
        print(f"writing annotated video: {args.save_video} ({w}x{h} @ {fps:.2f} fps)")

    try:
        if src is not None:
            n_total = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
            import time
            t_start = time.time()
            n = 0
            near_tracker = NearTracker(persist=3)   # 3-frame persistence
            while True:
                ok, img = src.read()
                if not ok:
                    break
                n += 1
                if n % 2 == 1:                       # measure every 2nd frame
                    res = measure_frame(model, maps, img,
                                        min_brightness=args.min_brightness,
                                        tracker=near_tracker)
                ann = annotate(img, res, traj)        # stamp on every frame
                if args.cam is not None:
                    print_result(res)
                else:
                    d = res.distances
                    cl = res.closest
                    nm = res.near_min
                    print(f"frame {n:5d}  L={d.get('L', float('nan')):7.2f} "
                          f"R={d.get('R', float('nan')):7.2f} "
                          f"B={d.get('B', float('nan')):7.2f}  "
                          f"MIN={cl if cl is None else round(cl, 3)}  "
                          f"NEAR={nm if nm is None else round(nm, 3)}")
                if writer is not None:
                    writer.write(ann)
                if args.cam is not None:
                    cv2.imshow("laser measure (q to quit)", ann)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if n % 100 == 0:
                    el = time.time() - t_start
                    eta = el / n * (n_total - n) if n_total else 0
                    cl = res.closest
                    print(f"frame {n}/{n_total}  min={cl if cl is None else round(cl, 2)} m   "
                          f"({el:.0f}s elapsed, ~{eta:.0f}s left)")
        else:
            img = cv2.imread(args.image)
            if img is None:
                raise SystemExit(f"cannot read {args.image}")
            res = measure_frame(model, maps, img,
                                min_brightness=args.min_brightness)
            print_result(res)
            out = args.out or os.path.splitext(args.image)[0] + ".annot.jpg"
            cv2.imwrite(out, annotate(img, res, traj))
            print(f"\nwrote {out}")
    finally:
        if src is not None:
            cv2.destroyAllWindows()
            src.release()
        if writer is not None:
            writer.release()
            print(f"\ndone: {args.save_video}")


if __name__ == "__main__":
    main()
