#!/usr/bin/env python3
"""
laser_radar_final.py — FINAL camera-online "radar matrix"
closest-obstacle measurement
============================================================

Camera-online build of laser_radar.py:
  * default mode is the USB camera (--cam 0);  NOTHING is written to
    disk --  everything is shown live in the window:  annotated
    camera view + 3-frame accumulated polar heat-map on the right
  * --out <file> stays available for occasional screencasts / tests
    (off by default)
  * 2 m distance floor,  5-px bright-segment object rule (a small
    object may intersect only one laser sheet,  even at 13 m),
    distance from the near face only,  3-frame persistence,
    static + live junk masks,  suspicious-direction reporting

Quick start
-----------
    python3 laser_radar_final.py                  # live camera (window only)
    python3 laser_radar_final.py --out my.mp4     # optional screencast recording
    python3 laser_radar_final.py --test           # 28-image validation

Idea
----
Instead of fitting laser lines per frame, the image is RESAMPLED into a
( angle x distance ) matrix, exactly like a radar:

  * columns = rotation angle, every 0.3 deg  (0..359.7, 1200 cols)
      measured from the horizontal, around the VANISHING CENTER C
      (the point where all laser traces collapse as d -> infinity,
      C ~= (614, 225) here;  note: "0 deg from the vertical line" in
      the original idea = 90 deg in this convention)
  * rows    = distance from the camera, 1 cm rows from 0.3 m to
      13.2 m, then 20 cm rows up to 40 m   (row 0 = closest)

A cell (theta, d) stores 0..3 = how many of the three laser sheets
(L / R / B) have their calibrated trace line at distance d crossing
the pixel sampled on the ray at angle theta:

      X_T(theta, d) =  ray from C at angle theta   intersect
                      the calibrated image line of sheet T at d

The calibrated line of sheet T at distance d comes straight from the
28-image calibration (LineMap:  x0(d), slope(d));  below the closest
calibration distance (1.955 m) and above the farthest (13.2 m) the
line is extrapolated with the local 1/d branch (slope ~ k/d^2, the
exact projective behaviour), so near objects down to 0.3 m work.

Per frame:
  1. threshold the frame (histogram rule), dilate 3x3
  2. sample the mask at X_T(theta, d) for all three sheets
       -> matrix M (n_d x n_theta)  via 3 cv2.remap calls
  3. row activity = number of lit cells per distance row
  4. NEAREST OBJECT = a bright SPOT that lies on one of the laser lines
       and spans >= SPOT_MIN_EXT (5) px ALONG that line  (a real object
       lights a run of the trace;  a line crossing is a 1-2 px flash).
       A small object may intersect only ONE sheet even at 13 m --
       no second-family confirmation is required at any distance.
       Its distance is measured ONLY from the (up to) SPOT_N_MEASURE
       pixels of the spot at the NEAREST distance (the near face) --
       a wide/thick spot never reports its middle or its far side.
  5. the per-column first lit cell gives the CONTOUR of the closest
       surface (the "real contour of the closest obstacle")
  6. a candidate must persist in 3 consecutive frames to be reported

Usage
-----
    python3 laser_radar_final.py --build-calib  # save vanishing center
    python3 laser_radar_final.py --test         # 28-image check
    python3 laser_radar_final.py               # live camera (auto-record)
    python3 laser_radar_final.py --video f.mp4 # process a video file
    python3 laser_radar_final.py --debug-polar # extra large radar window
Keys: q / ESC to quit (the recording is finalized on exit).
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laser_sensor as ls

REF_W, REF_H = 1280, 720
FAMILIES = ("L", "R", "B")

# ------------------------- tuning -------------------------------------
THETA_STEP = 0.3          # deg per column  (user's "every 0.3 degree")
D_MIN = 2.00              # closest row, m  (distances < 2 m are out of
                         # range:  the corresponding image region is
                         # excluded from the very start)
D_MAX = 40.0              # farthest row, m
NEAR_STEP = 0.02          # row resolution inside calibration range, m
FAR_STEP = 1.0            # row resolution beyond calibration range, m
MIN_ROW_ACT = 3           # min lit cells in a row to count it as a surface
WALL_ACT_MIN = 200        # peak streak-weighted score for "the wall"
STREAK_MIN = 8            # consecutive angle columns = a real streak
D_CAL_MIN = 1.955         # closest calibration distance (m)
D_CAL_MAX = 13.20         # farthest calibration distance (m)

# ---- nearest-object (spot) rule ----------------------------------------
SPOT_MIN_AREA = 10       # px,  min area of a bright component
SPOT_MIN_EXT = 5         # px,  spot must span >= this ALONG its laser line
SPOT_N_MEASURE = 20       # px,  ONLY these nearest-distance pixels are used
                          #      for the distance (never the spot's middle
                          #      or far side)
SPOT_MAX_OFF = 5.0        # px,  max median pixel-to-line distance = "on line"
SPOT_JUNK_FRAC = 0.5      # skip a spot if > this fraction of its selected
                          #      pixels falls into junk cells (static scene)
SPOT_WALL_MARGIN = 0.25   # m,  a near object must sit this far in front
                          #      of the wall  (the wall itself is reported
                          #      separately,  not as a near object)
SPOT_MAX_COMPS = 20       # max components examined per frame (largest first)
PERSIST = 3               # frames of persistence before reporting
FAM_COLOR = {"L": (255, 191, 0), "R": (0, 191, 255), "B": (0, 255, 255)}


# ------------------------- primitives ---------------------------------

def laser_mask(gray: np.ndarray,
               min_brightness: Optional[float] = None) -> np.ndarray:
    """Mask of thin bright structures (laser traces / bright edges).

    Top-hat against a 31-px local background:  a pixel passes only if
    it is clearly brighter than its neighborhood,  so flat bright
    regions (windows,  lit walls) vanish and only locally bright lines
    remain.  Works both in dark rooms (traces on a black scene) and in
    bright rooms (traces on a lit wall).  A flat region's thin frame
    survives the top-hat but is static,  so the junk masks kill it."""
    if min_brightness is not None:
        return (gray > min_brightness).astype(np.uint8)
    local = cv2.subtract(gray, cv2.blur(gray, (31, 31)))
    return (local > 25).astype(np.uint8)


def _line_coeffs(name: str, m_val: float, x0_val: float):
    """Normalized line (a,b,c) of a sheet trace at a given distance.
    L/R: x = m*(y-360) + x0 ;  B: y = m*(x-640) + x0."""
    if name == "B":
        a, b, c = -m_val, 1.0, m_val * 640.0 - x0_val
    else:
        a, b, c = 1.0, -m_val, m_val * 360.0 - x0_val
    n = math.hypot(a, b)
    return a / n, b / n, c / n


def _fit_1d(d: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Least-squares fit  y = v0 + k/d ."""
    A = np.vstack([np.ones_like(d), 1.0 / d]).T
    return tuple(np.linalg.lstsq(A, y, rcond=None)[0])


# ------------------------- fine line map --------------------------------

class FineLine:
    """Trace line of one sheet at ANY distance d in [0.25, 45].

    inside the calibration range [lo, hi]:  node interpolation (the
    measured truth, incl. R fan wiggles);
    below lo:  1/d branch fitted to the 3 closest nodes (local
        projective slope ~ k/d^2), blended over 0.15 m;
    above hi:  1/d branch fitted to the 3 farthest nodes, blended.
    Pre-sampled on a 2.5 mm d grid for O(1) lookup."""

    def __init__(self, name: str, mp: ls.LineMap):
        self.name = name
        self.ds = np.asarray(mp.ds, float)
        self.x0s = np.asarray(mp.x0s, float)
        self.ms = np.asarray(mp.ms, float)
        self.lo, self.hi = float(self.ds[0]), float(self.ds[-1])
        (self.nv0, self.nk), (self.nvm, self.nkm) = (
            _fit_1d(self.ds[:3], self.x0s[:3]),
            _fit_1d(self.ds[:3], self.ms[:3]))
        (self.fv0, self.fk), (self.fvm, self.fkm) = (
            _fit_1d(self.ds[-3:], self.x0s[-3:]),
            _fit_1d(self.ds[-3:], self.ms[-3:]))
        # pre-sample
        dg = np.unique(np.concatenate([
            np.arange(0.25, self.lo, 0.0025),
            self.ds,
            np.arange(self.hi + 0.025, 45.0, 0.025)]))
        a = np.empty(len(dg)); b = np.empty(len(dg)); c = np.empty(len(dg))
        x0 = np.empty(len(dg)); sl = np.empty(len(dg))
        for i, d in enumerate(dg):
            if d < self.lo:
                w = min(max((d - (self.lo - 0.15)) / 0.15, 0.0), 1.0)
                x0n = self.nv0 + self.nk / d
                sln = self.nvm + self.nkm / d
                x0[i] = (1 - w) * x0n + w * self.x0s[0]
                sl[i] = (1 - w) * sln + w * self.ms[0]
            elif d <= self.hi:
                x0[i] = np.interp(d, self.ds, self.x0s)
                sl[i] = np.interp(d, self.ds, self.ms)
            else:
                w = min(max((d - (self.hi + 0.15)) / 0.15, 0.0), 1.0)
                x0f = self.fv0 + self.fk / d
                slf = self.fvm + self.fkm / d
                x0[i] = (1 - w) * self.x0s[-1] + w * x0f
                sl[i] = (1 - w) * self.ms[-1] + w * slf
            a[i], b[i], c[i] = _line_coeffs(self.name, sl[i], x0[i])
        self.dg = dg
        self.a, self.b, self.c = a, b, c
        # uniform ~18-cm grid for the per-pixel d mapping in the spot
        # detector (argmin over this grid per pixel;  the 18 cm step is
        # small enough that any true line is within the 5 px match
        # window,  yet coarse enough to stay fast)
        self.gd = np.arange(0.5, 45.0, 0.18)
        self.ga, self.gb, self.gc = self.coeffs(self.gd)

    def coeffs(self, d_arr: np.ndarray):
        """(a,b,c) arrays for an array of distances (interpolated;
        a,b,c are linear in d between grid nodes,  so linear
        interpolation is exact there and sub-pixel elsewhere)."""
        d_arr = np.clip(np.asarray(d_arr, float),
                        float(self.dg[0]), float(self.dg[-1]))
        return (np.interp(d_arr, self.dg, self.a),
                np.interp(d_arr, self.dg, self.b),
                np.interp(d_arr, self.dg, self.c))


# ------------------------- vanishing center -----------------------------

def vanishing_center(maps: Dict[str, ls.LineMap]) -> np.ndarray:
    """C = average of the L x B and R x B asymptotic-line intersections
    (d -> infinity)."""
    asym = {}
    for name, mp in maps.items():
        ds = np.asarray(mp.ds, float)
        (v0, k), (vm, km) = (_fit_1d(ds, np.asarray(mp.x0s, float)),
                             _fit_1d(ds, np.asarray(mp.ms, float)))
        a, b, c = _line_coeffs(name, vm, v0)
        asym[name] = (a, b, c)
    pts = []
    for n1, n2 in (("L", "B"), ("R", "B")):
        A = np.array([asym[n1][:2], asym[n2][:2]])
        p = np.linalg.solve(A, -np.array([asym[n1][2], asym[n2][2]]))
        if -200 <= p[0] <= REF_W + 200 and -200 <= p[1] <= REF_H + 200:
            pts.append(p)
    return np.mean(pts, axis=0) if pts else np.array([614.0, 225.0])


# ------------------------- radar matrix ---------------------------------

class RadarMatrix:
    """Precomputed remap maps: for every (d row, theta col) the image
    pixel X_T(theta, d) = ray(C, theta) ∩ calibrated line of sheet T at d.
    Invalid intersections (behind C / off-image / parallel) are parked
    outside the frame so remap returns 0 (border)."""

    def __init__(self, fine: Dict[str, FineLine], C: np.ndarray,
                 theta_step: float = THETA_STEP, dmin: float = D_MIN,
                 dmax: float = D_MAX, near_step: float = NEAR_STEP,
                 far_step: float = FAR_STEP):
        self.C = np.asarray(C, float)
        n_th = int(round(360.0 / theta_step))
        self.theta = np.arange(n_th) * theta_step
        # rows: 1 cm up to the calibration range end, 20 cm beyond
        hi = max(f.hi for f in fine.values())
        rows_near = np.arange(dmin, hi + 1e-9, near_step)
        rows_far = np.arange(hi + near_step, dmax + 1e-9, far_step)
        self.d = np.concatenate([rows_near, rows_far])
        self.n_th, self.n_d = n_th, len(self.d)
        th = np.deg2rad(self.theta)
        ux, uy = np.cos(th), np.sin(th)
        cx, cy = self.C[0], self.C[1]
        self.map_x: Dict[str, np.ndarray] = {}
        self.map_y: Dict[str, np.ndarray] = {}
        self.pts: Dict[str, np.ndarray] = {}
        for T, fl in fine.items():
            A, B, Cc = fl.coeffs(self.d)
            den = A[:, None] * ux[None, :] + B[:, None] * uy[None, :]
            num = -(A[:, None] * cx + B[:, None] * cy + Cc[:, None])
            with np.errstate(divide="ignore", invalid="ignore"):
                t = num / den
            X = cx + t * ux[None, :]
            Y = cy + t * uy[None, :]
            bad = (~np.isfinite(t)) | (t < 1.0) | (X < 0) | (X > REF_W - 1) | \
                  (Y < 0) | (Y > REF_H - 1) | (np.abs(den) < 1e-9)
            X[bad] = -5.0
            Y[bad] = -5.0
            self.map_x[T] = X.astype(np.float32)
            self.map_y[T] = Y.astype(np.float32)
            self.pts[T] = np.stack([X, Y], axis=-1)


def sample_matrix(mask: np.ndarray, rmat: RadarMatrix) -> Tuple[np.ndarray,
                                                                Dict[str, np.ndarray]]:
    """Sample a (dilated) bright mask at all X_T(theta, d).
    Returns M (n_d x n_th, 0..3) and the per-family sample arrays."""
    S = {}
    M = np.zeros((rmat.n_d, rmat.n_th), dtype=np.uint8)
    for T in FAMILIES:
        s = cv2.remap(mask, rmat.map_x[T], rmat.map_y[T],
                      cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=0)
        S[T] = s
        M += s
    return M.clip(0, 3), S


# ------------------------- static-junk mask ------------------------------

def build_junk_mask(fine: Dict[str, FineLine], rmat: RadarMatrix,
                    cal_dir: str, n_static: int = 15,
                    out_path: str = "radar_junk.npy") -> np.ndarray:
    """Cells that are lit in >= n_static of the 28 calibration images are
    STATIC room structures (corner edges, floor reflections, lamps...) --
    they map to fixed (theta, d) cells because the (theta, d) -> pixel
    mapping is fixed.  A real object only appears in a live frame, so it
    is never part of this mask.  The mask is zeroed out of M per frame.
    The laser traces themselves are NOT static in (theta, d) (they move
    with the target distance), so they survive the mask."""
    imgs, _ = _load_calib_images(cal_dir)
    acc = np.zeros((rmat.n_d, rmat.n_th), dtype=np.int32)
    n_img = 0
    for path in imgs:
        img = cv2.imread(path)
        img = cv2.resize(img, (REF_W, REF_H))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = cv2.dilate(laser_mask(gray), np.ones((3, 3), np.uint8))
        M, _ = sample_matrix(mask, rmat)
        acc += (M > 0)
        n_img += 1
    junk = acc >= n_static
    np.save(out_path, junk)
    print(f"junk mask: {junk.sum()} cells static "
          f"({100.0 * junk.sum() / junk.size:.1f}% of the matrix), "
          f"lit in >= {n_static}/{n_img} calib images -> saved "
          f"{os.path.basename(out_path)}")
    return junk


def load_junk_mask(path: str) -> Optional[np.ndarray]:
    if os.path.exists(path):
        return np.load(path)
    return None


# ------------------------- nearest-object spot detector -----------------

def spot_candidates(mask: np.ndarray, fine: Dict[str, FineLine],
                    rmat: RadarMatrix, junk: Optional[np.ndarray] = None,
                    d_cap: Optional[float] = None,
                    d_max: float = D_CAL_MAX,
                    gray: Optional[np.ndarray] = None) -> List[dict]:
    """All bright spots on a laser line spanning >= SPOT_MIN_EXT (5) px
    ALONG that line,  with the distance measured from the NEAREST
    (up to SPOT_N_MEASURE) pixels of the spot (the near face) --  a
    wide or thick spot never reports its middle or its far side.
    A single family is enough at any distance:  a small object can
    intersect only one sheet even at 13 m.

    Per component and per family:
      * every pixel is mapped to the distance d_p of the family's
        calibrated line that passes closest to it  (argmin over the 4-cm
        grid:  f(d) = a(d)x + b(d)y + c(d) = 0)
      * the component is "on the line" if,  for some d,  a solid
        fraction of its pixels lie within SPOT_MAX_OFF of line_T(d)
      * the extent of those on-line pixels ALONG the line must be >=
        SPOT_MIN_EXT (a crossing edge is only a few px,  an object
        lights a long run of the trace)
      * selected pixels are rejected if they map mostly into junk cells
        (static scene)  or lie at/behind the wall (d_cap)
    Returns the nearest (smallest d) candidate,  or None.
    """
    ncomp, lab, stats, _cents = cv2.connectedComponentsWithStats(mask, 8)
    if ncomp <= 1:
        return []
    ys, xs = np.nonzero(lab)
    if not len(xs):
        return []
    labv = lab[ys, xs]
    order = np.argsort(labv, kind="stable")
    labv = labv[order]
    ys, xs = ys[order], xs[order]
    starts = np.unique(labv, return_index=True)[1]
    # qualifying components,  largest first (bounded work)
    qual = sorted(((int(stats[cid, cv2.CC_STAT_AREA]), cid)
                  for cid in range(1, ncomp)
                  if int(stats[cid, cv2.CC_STAT_AREA]) >= SPOT_MIN_AREA),
                 reverse=True)[:SPOT_MAX_COMPS]
    cands = []
    cx0, cy0 = float(rmat.C[0]), float(rmat.C[1])
    for area, cid in qual:
        s0 = int(starts[cid - 1])
        s1 = int(starts[cid]) if cid < ncomp - 1 else len(xs)
        n = s1 - s0
        step = max(1, n // 128)
        idx = np.arange(0, n, step)
        X = xs[s0 + idx].astype(np.float32)
        Y = ys[s0 + idx].astype(np.float32)
        for T in FAMILIES:
            gd, ga, gb, gc = (fine[T].gd, fine[T].ga, fine[T].gb,
                              fine[T].gc)
            f = ga[:, None] * X[None, :] + gb[:, None] * Y[None, :] + \
                gc[:, None]
            ad = np.abs(f)
            d_p = gd[ad.argmin(axis=0)]          # per-pixel nearest d
            # is the component ON this family's line at some d ?
            cnt = (ad <= SPOT_MAX_OFF).sum(axis=1)
            i_best = int(cnt.argmax())
            if cnt[i_best] < max(5, int(0.4 * len(X))):
                continue
            d_star = float(gd[i_best])
            (a2,), (b2,), (c2,) = fine[T].coeffs(np.array([d_star]))
            a2, b2, c2 = float(a2), float(b2), float(c2)
            res = np.abs(a2 * X + b2 * Y + c2)
            on = res <= 2.0 * SPOT_MAX_OFF
            if int(on.sum()) < 5:
                on = res <= SPOT_MAX_OFF
                if int(on.sum()) < 5:
                    continue
            # extent of the on-line pixels ALONG the line direction
            ux, uy = -b2, a2
            proj = X[on] * ux + Y[on] * uy
            ext = float(proj.max() - proj.min())
            if ext < SPOT_MIN_EXT:
                continue
            # the distance:  ONLY the nearest SPOT_N_MEASURE pixels
            order_d = np.argsort(d_p[on])
            k = min(SPOT_N_MEASURE, int(on.sum()))
            sel = order_d[:k]
            d_cand = float(np.median(d_p[on][sel]))
            if d_cand < D_MIN or d_cand > d_max:
                continue                      # outside the radar range
            if d_cap is not None and d_cand > d_cap:
                continue                      # the wall itself / behind it
            # brightness gate:  a real laser trace is the BRIGHTEST
            # thing in the frame (saturated);  local-contrast room
            # features (shadows,  dark bands,  wall texture,  window
            # edges) that the top-hat picks up are dim.  Require at
            # least half of the 30 nearest on-line pixels to sit at or
            # above the frame's bright threshold.
            if gray is not None:
                # p99 on a 1/4 subsample:  ~4x faster,  same estimate
                thr_b = max(200.0,
                            float(np.percentile(gray[::4, ::4], 99)) - 25.0)
                gsel = np.asarray(
                    gray[Y[on][sel].astype(int), X[on][sel].astype(int)],
                    np.float32)
                if int((gsel >= thr_b).sum()) < max(3, k // 2):
                    continue
            sx = float(X[on][sel].mean())
            sy = float(Y[on][sel].mean())
            theta = math.degrees(math.atan2(sy - cy0, sx - cx0)) % 360.0
            if junk is not None:
                th_p = np.degrees(np.arctan2(Y[on][sel] - cy0,
                                             X[on][sel] - cx0)) % 360.0
                ii = np.clip(np.searchsorted(rmat.d, d_p[on][sel]),
                             0, rmat.n_d - 1)
                jj = np.clip((th_p / THETA_STEP).astype(np.int64),
                             0, rmat.n_th - 1)
                if int(junk[ii, jj].sum()) > SPOT_JUNK_FRAC * k:
                    continue                  # static scene feature
            cands.append({"d": d_cand, "theta": theta, "fam": T,
                          "xy": (sx, sy), "n": k, "ext": ext})
    return cands


def pick_nearest_spot(cands: List[dict]) -> Optional[dict]:
    """Nearest-candidate selection.

    A small object can intersect only ONE laser sheet even at 13 m,
    so a candidate is accepted on a single family at ANY distance.
    Protection against static phantoms lives in the other layers:
    the static / live junk masks,  the brightness gate,  the d_max
    calibration cap,  and the 3-frame persistence.
    Returns the nearest candidate,  or None.
    """
    if not cands:
        return None
    return min(cands, key=lambda c: c["d"])


# ------------------------- frame analysis --------------------------------

def _circular_mean(theta: np.ndarray, w: np.ndarray) -> float:
    v = np.exp(1j * np.deg2rad(theta))
    ang = np.angle(np.average(v, weights=w))
    return float(np.degrees(ang)) % 360.0


def _max_streak(circ: np.ndarray) -> int:
    """Longest run of 1s on a circular boolean array."""
    n = len(circ)
    if not circ.any():
        return 0
    best = 0
    doubled = np.concatenate([circ, circ])
    run = 0
    # find a zero to start counting from
    z = np.where(~doubled)[0]
    if len(z) == 0:
        return n
    start = z[0] + 1
    if start >= 2 * n:
        start = 0
    for i in range(start, start + n):
        if doubled[i]:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


class TemporalFilter:
    """Accumulation over the last N (3) consecutive frames.

    A cell is STABLE only if it was active in ALL of the last three
    consecutive frames;  the accumulated count (0..3) is what the
    heat-map shows (1/3 dim,  2/3 medium,  3/3 bright) --  single-frame
    flicker vanishes,  moving objects leave a short 3-frame trail.
    """

    DIL = np.ones((3, 3), np.uint8)   # jitter tolerance (handheld shake)

    def __init__(self, n: int = 3):
        self.n = n
        self.buf: "deque" = deque(maxlen=n)

    def update(self, M: np.ndarray, S: Dict[str, np.ndarray]):
        self.buf.append((M, S))
        if len(self.buf) < self.n:
            return None
        acc_M = None
        acc_S = {T: None for T in FAMILIES}
        for M_, S_ in self.buf:
            # 3x3 dilation:  a +-1..2 px shake between frames must not
            # break the "lit in all 3 frames" test
            m = cv2.dilate((M_ > 0).astype(np.uint8), self.DIL)
            acc_M = m if acc_M is None else acc_M + m
            for T in FAMILIES:
                s = cv2.dilate((S_[T] > 0).astype(np.uint8), self.DIL)
                acc_S[T] = s if acc_S[T] is None else acc_S[T] + s
        return acc_M, {T: (acc_S[T] == self.n).astype(np.uint8)
                        for T in FAMILIES}

    def reset(self):
        self.buf.clear()


def radar_frame(fine: Dict[str, FineLine], rmat: RadarMatrix,
                gray: np.ndarray, min_brightness: Optional[float] = None,
                min_row: int = MIN_ROW_ACT,
                junk: Optional[np.ndarray] = None,
                allow_sub_cal: bool = False,
                d_max: float = D_CAL_MAX,
                hist: Optional[TemporalFilter] = None) -> dict:
    mask = cv2.dilate(laser_mask(gray, min_brightness),
                      np.ones((3, 3), np.uint8))
    M, S = sample_matrix(mask, rmat)
    if junk is not None:
        M &= ~junk

    # ---- 3-frame temporal accumulation --------------------------------
    # a cell is STABLE only if it was active in ALL of the last three
    # consecutive frames;  wall / contour / suspicious are computed on
    # the stable set (no single-frame flicker),  and the accumulated
    # count (0..3) is what the heat-map shows.
    acc_M = None
    stable_S = None
    if hist is not None:
        r = hist.update(M, S)
        if r is not None:
            acc_M, stable_S = r
    use_M = ((acc_M == 3) if acc_M is not None else (M > 0)).astype(np.uint8)
    use_S = {T: (stable_S[T] if stable_S is not None
                 else (S[T] > 0).astype(np.uint8)) for T in FAMILIES}

    d = rmat.d
    row_act = M.sum(axis=1, dtype=np.int32)
    lo_ok = int(np.searchsorted(d, D_CAL_MIN))     # first measured row
    hi_ok = int(np.searchsorted(d, D_CAL_MAX + 0.5))

    # ---- wall row: streak-weighted score per distance row --------------
    # a real wall lights long angle streaks in several sheets;  room junk
    # (edges, target-board trajectories) lights only short segments, so
    # each family's contribution is weighted by its eroded (run>=8)
    # support, not just by cell count.  The peak is searched only inside
    # the measured range (the rows beyond it are extrapolated).
    score = np.zeros(rmat.n_d, dtype=np.float32)
    fam_cells = np.zeros((len(FAMILIES), rmat.n_d), dtype=np.int32)
    for k, T in enumerate(FAMILIES):
        b = use_S[T]
        er = cv2.erode(b, np.ones((1, STREAK_MIN), np.uint8))
        score += (0.3 * b + 0.7 * er).sum(axis=1)
        fam_cells[k] = b.sum(axis=1)
    # a wall lights several sheets at once;  single-sheet structures
    # (target-board edges, room corners) are discounted 10x
    fams_present = (fam_cells >= 8).sum(axis=0)
    cand = score * np.where(fams_present >= 2, 1.0, 0.1)
    cand[:lo_ok] = 0
    cand[hi_ok:] = 0
    d_wall = None
    if cand.max() >= WALL_ACT_MIN:
        peak = int(np.argmax(cand))
        band = (np.abs(d - d[peak]) <= 0.5) & (cand >= 0.4 * cand[peak])
        d_wall = float(np.average(d[band], weights=cand[band]))

    # ---- nearest object:  a bright spot on a laser line ---------------
    # user's rule:  a bright segment spanning >= SPOT_MIN_EXT (5) px
    # along its laser line is an object  (a small object may intersect
    # only one sheet,  even at 13 m);  its distance is measured ONLY
    # from the (up to) SPOT_N_MEASURE pixels of the spot at the nearest
    # distance (the near face) --  never from the middle or far side.
    d_near = theta_near = None
    n_cells = 0
    streak = 0
    fam_near = None
    near_xy = None
    d_cap = (d_wall - SPOT_WALL_MARGIN) if d_wall is not None else None
    cands = spot_candidates(mask, fine, rmat, junk,
                            d_cap=d_cap, d_max=d_max, gray=gray)
    spot = pick_nearest_spot(cands)
    if spot is not None:
        d_near = spot["d"]
        theta_near = spot["theta"]
        fam_near = spot["fam"]
        near_xy = spot["xy"]
        n_cells = spot["n"]
        streak = int(spot["ext"])

    # ---- contour: first lit cell of every angle column -----------------
    jmax = int(np.searchsorted(d, (d_wall + 0.3) if d_wall is not None
                               else D_MAX + 1))
    jmax = min(jmax, rmat.n_d)
    Mb = use_M[:jmax, :]
    first = Mb.argmax(axis=0)
    has = Mb.any(axis=0)
    contour: List[Tuple[float, float, float, float, str]] = []
    if has.any():
        i_idx = first[has]
        c_idx = np.where(has)[0]
        vals = np.stack([use_S[T][i_idx, c_idx] for T in FAMILIES], axis=0)
        fam_idx = vals.argmax(axis=0)
        x = np.empty(len(i_idx), dtype=np.float32)
        y = np.empty(len(i_idx), dtype=np.float32)
        for k, T in enumerate(FAMILIES):
            m = fam_idx == k
            if m.any():
                x[m] = rmat.pts[T][i_idx[m], c_idx[m], 0]
                y[m] = rmat.pts[T][i_idx[m], c_idx[m], 1]
        contour = list(zip(rmat.theta[has].tolist(), d[i_idx].tolist(),
                           x.tolist(), y.tolist(),
                           [FAMILIES[int(k)] for k in fam_idx]))

    # ---- suspicious directions:  no bright spot anywhere --------------
    # a direction with NO lit cell at all means the light in that
    # direction either went to infinity (converged to C without a
    # visible hit) or the surface did not reflect the beam back --
    # no distance can be trusted there,  so the direction is flagged.
    # A column fully covered by the static-junk mask is KNOWN scene
    # (a static wall),  not suspicious.
    col_act = use_M.sum(axis=0, dtype=np.int32)
    susp_col = (col_act == 0)
    if junk is not None:
        junk_frac = junk.sum(axis=0) / float(rmat.n_d)
        susp_col &= (junk_frac < 0.9)
    susp: List[Tuple[float, float]] = []
    run_start = None
    for j in range(rmat.n_th + 1):
        is_zero = (j < rmat.n_th) and bool(susp_col[j])
        if is_zero and run_start is None:
            run_start = j
        elif not is_zero and run_start is not None:
            if j - run_start >= 2:      # min 2 cols = 0.6 deg
                susp.append((float(rmat.theta[run_start]),
                             float(rmat.theta[j - 1])))
            run_start = None

    return {"M": M, "S": S, "row_act": row_act, "acc": acc_M,
            "d_wall": d_wall, "d_near": d_near, "theta_near": theta_near,
            "n_cells": n_cells, "streak": streak, "fam_near": fam_near,
            "near_xy": near_xy, "contour": contour,
            "susp": susp, "susp_col": susp_col}


# ------------------------- persistence -----------------------------------

class NearTracker:
    """A near candidate must stay (within 0.3 m / 4 deg) for 3 frames."""

    def __init__(self, persist: int = PERSIST):
        self.persist = persist
        self.n = 0
        self.d = None
        self.theta = None

    def reset(self):
        self.n = 0
        self.d = self.theta = None

    def update(self, d: Optional[float], theta: Optional[float]) -> bool:
        if d is None or theta is None:
            self.n = 0
            self.d = self.theta = None
            return False
        if self.n > 0:
            dd = abs(d - self.d)
            dth = abs(theta - self.theta)
            dth = min(dth, 360.0 - dth)
            if dd <= 0.3 and dth <= 4.0:
                self.n += 1
                self.d = 0.8 * self.d + 0.2 * d
                self.theta = theta
            else:
                self.n = 1
                self.d, self.theta = d, theta
        else:
            self.n = 1
            self.d, self.theta = d, theta
        return self.n >= self.persist


# ------------------------- annotation ------------------------------------

def annotate(img: np.ndarray, res: dict, rmat: RadarMatrix,
             reported: bool, fps: float, cpu_ms: float) -> np.ndarray:
    out = img.copy()
    d_wall = res["d_wall"]
    if d_wall is not None:
        cv2.putText(out, f"wall {d_wall:.2f} m", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    # (the per-angle contour outline is intentionally NOT drawn:
    #  it blinks frame to frame and distracts;  the polar heat-map on
    #  the right shows the same information stably)
    # nearest object
    if res["d_near"] is not None and res["near_xy"] is not None:
        cx_, cy_ = res["near_xy"]
        if reported:
            cv2.circle(out, (int(cx_), int(cy_)), 10, (0, 0, 255), 3)
            cv2.putText(out, f"MIN {res['d_near']:.2f} m",
                        (int(cx_) + 15, int(cy_) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
            cv2.putText(out, f"theta {res['theta_near']:.0f} deg  "
                        f"sheet {res['fam_near']}",
                        (int(cx_) + 15, int(cy_) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.circle(out, (int(cx_), int(cy_)), 5, (128, 128, 128), 2)
    # suspicious directions:  purple arcs -- no bright spot at all in
    # that direction (light went to infinity or the surface did not
    # reflect the beam back)
    W_, H_ = out.shape[1], out.shape[0]
    for (t0, t1) in res.get("susp", [])[:8]:
        for tdeg in np.linspace(t0, t1, 24):
            a = math.radians(tdeg)
            x = rmat.C[0] + 235.0 * math.cos(a)
            y = rmat.C[1] + 235.0 * math.sin(a)
            if 0 <= x < W_ and 0 <= y < H_:
                cv2.circle(out, (int(x), int(y)), 2, (160, 0, 160), -1)
    susp_list = res.get("susp", [])
    if susp_list:
        txt = " ".join(f"{a:.0f}-{b:.0f}" for a, b in susp_list[:3])
        extra = f" +{len(susp_list) - 3} more" if len(susp_list) > 3 else ""
        cv2.putText(out, f"SUSPICIOUS (no reflection): {txt}{extra}",
                    (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (160, 0, 160), 2)
    cv2.drawMarker(out, (int(rmat.C[0]), int(rmat.C[1])), (255, 255, 255),
                   cv2.MARKER_CROSS, 12)
    cv2.putText(out, f"fps {fps:5.1f}  cpu {cpu_ms:5.1f} ms   "
                     f"(rows={rmat.n_d} cols={rmat.n_th})",
                (20, REF_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 2)
    return out


def polar_debug_view(M: np.ndarray, rmat: RadarMatrix,
                     susp_col: Optional[np.ndarray] = None,
                     acc: Optional[np.ndarray] = None) -> np.ndarray:
    """The radar screen: rows = distance (0.3 m on top), cols = angle.
    If `acc` is given it shows the 3-frame accumulation (1/3 dim ..
    3/3 bright);  purple columns = suspicious directions (no bright
    spot at all in the last 3 consecutive frames)."""
    src_im = (acc * 85) // 3 if acc is not None else M.astype(np.uint8) * 85
    img = cv2.cvtColor(
        cv2.resize(src_im.astype(np.uint8), (360, 260),
                   interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR)
    for i in range(0, rmat.n_d, max(1, rmat.n_d // 10)):
        y = int(i / rmat.n_d * 260)
        dd = rmat.d[i]
        if dd >= 1:
            cv2.putText(img, f"{dd:.0f}m", (2, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    for t in (0, 90, 180, 270):
        x = int(t / 360 * 360)
        cv2.line(img, (x, 0), (x, 259), (120, 120, 120), 1)
        cv2.putText(img, f"{t}", (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1)
    if susp_col is not None:
        col = np.array([150, 40, 150], np.uint8)[None, None, :]
        for j in np.where(susp_col)[0]:
            x = int(float(rmat.theta[j]) / 360 * 360)
            img[:, max(0, x - 1):x + 2] = col
    return img


# ------------------------- camera / video --------------------------------

def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        devs = [f"/dev/video{i}" for i in range(8)
                if os.path.exists(f"/dev/video{i}")]
        print(f"cannot open camera {index};  found devices: {devs}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"camera {index}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


def probe_camera_rate(cap: cv2.VideoCapture, seconds: float = 2.0) -> float:
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    return n / max(time.time() - t0, 1e-6)


# ------------------------- modes -----------------------------------------

def _load_calib_images(cal_dir: str):
    import openpyxl
    imgs = sorted(glob.glob(os.path.join(cal_dir, "WIN_*.jpg")))
    wb = openpyxl.load_workbook(os.path.join(cal_dir,
                                             "camera_calibration.xlsx"))
    ws = wb.active
    dists = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row and row[1] is not None:
            try:
                dists.append(float(row[1]))
            except (TypeError, ValueError):
                pass
    assert len(dists) == len(imgs), \
        f"{len(dists)} distances vs {len(imgs)} images"
    return imgs, dists


def cmd_test(fine, rmat, cal_dir: str, junk, ascii_view: bool = False):
    imgs, dists = _load_calib_images(cal_dir)
    print(f"{'true d':>8} {'wall':>7} {'w err':>7}  {'near':>7} {'n err':>7}  "
          f"{'px':>3} {'ext':>4} {'susp deg':>8}")
    werrs, nerrs = [], []
    n_in = 0
    spurious = 0
    for k, (path, dj) in enumerate(zip(imgs, dists)):
        img = cv2.imread(path)
        img = cv2.resize(img, (REF_W, REF_H))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = radar_frame(fine, rmat, gray, junk=junk)
        w = res["d_wall"]
        nn = res["d_near"]
        wtxt = f"{w:7.3f}" if w is not None else "   NONE "
        ntxt = f"{nn:7.3f}" if nn is not None else "   --  "
        if dj < D_MIN:
            # target board below the radar floor:  out of range by design.
            #  (it is the nearest surface,  so it is reported as the wall
            #  clamped at the floor or not at all)
            if nn is not None:
                spurious += 1
            nsusp = int(sum(b - a for a, b in res.get("susp", [])))
            print(f"{dj:8.3f} {wtxt}  OUT(<{D_MIN:.0f}m)  {ntxt} "
                  f"{'   -- ':>7}  {res['n_cells']:3d} {res['streak']:4d} "
                  f"{nsusp:8d}")
            continue
        n_in += 1
        # the target board IS the nearest surface in these images,  so it
        # is reported as the wall and there must be NO near object
        werr = f"{w - dj:+7.3f}" if w is not None else "   --  "
        if w is not None:
            werrs.append(w - dj)
        nerr = f"{nn - dj:+7.3f}" if nn is not None else "   --  "
        if nn is not None:
            spurious += 1
            nerrs.append(nn - dj)
        nsusp = int(sum(b - a for a, b in res.get("susp", [])))
        print(f"{dj:8.3f} {wtxt} {werr}  {ntxt} {nerr}  "
              f"{res['n_cells']:3d} {res['streak']:4d} {nsusp:8d}")
        if ascii_view and k in (0, len(imgs) // 2, len(imgs) - 1):
            _print_ascii(res["M"], rmat)
    werrs = np.array([e for e in werrs if np.isfinite(e)])
    nerrs = np.array([e for e in nerrs if np.isfinite(e)])
    if len(werrs):
        print(f"\nwall (=board face):  RMS = {np.sqrt((werrs**2).mean()):.3f} m,  "
              f"max = {np.abs(werrs).max():.3f} m  over {len(werrs)} images")
    if len(nerrs):
        print(f"near RMS = {np.sqrt((nerrs**2).mean()):.3f} m,  "
              f"max = {np.abs(nerrs).max():.3f} m  over {len(nerrs)} images")
    print(f"spurious near objects on flat-board images: {spurious}")


def _print_ascii(M: np.ndarray, rmat: RadarMatrix):
    """Crude radar screen: 60 cols x 26 rows, row 0 = 0.3 m on top."""
    h, w = M.shape
    rh, rw = 26, 60
    small = cv2.resize(M.astype(np.float32), (rw, rh),
                       interpolation=cv2.INTER_AREA)
    chars = " .:-=+*#%@"
    print("  angle: 0" + " " * 54 + "360")
    for i in range(rh):
        dd = rmat.d[min(int(i / rh * h), h - 1)]
        line = "".join(chars[min(int(v * 9), 9)] for v in small[i])
        print(f"  {dd:5.1f}m|{line}|")


def cmd_build_calib(maps: Dict[str, ls.LineMap], out_path: str):
    C = vanishing_center(maps)
    data = {"center": [float(C[0]), float(C[1])], "theta_step": THETA_STEP,
            "dmin": D_MIN, "dmax": D_MAX, "near_step": NEAR_STEP,
            "far_step": FAR_STEP}
    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"vanishing center C = ({C[0]:.1f}, {C[1]:.1f})")
    print(f"saved {out_path}")
    return C


def main():
    ap = argparse.ArgumentParser(description="radar-matrix laser measurement")
    ap.add_argument("--cal", default="calibration.json")
    ap.add_argument("--rcal", default="radar_calibration.json")
    ap.add_argument("--cal-dir",
                    default=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "calibration"))
    ap.add_argument("--build-calib", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--ascii", action="store_true",
                    help="print ASCII radar screens in --test mode")
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=REF_W)
    ap.add_argument("--height", type=int, default=REF_H)
    ap.add_argument("--min-brightness", type=float, default=None)
    ap.add_argument("--min-row", type=int, default=MIN_ROW_ACT)
    ap.add_argument("--sub-cal", action="store_true",
                    help="also allow near candidates below the closest "
                         "calibration distance (experimental)")
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--no-live-mask", action="store_true",
                    help="do not learn a live-scene static junk mask")
    ap.add_argument("--boot-frames", type=int, default=120,
                    help="frames used to learn the live-scene junk mask")
    ap.add_argument("--live-fraction", type=float, default=0.25,
                    help="a cell lit in >= this fraction of boot frames "
                         "is static scene and gets masked")
    ap.add_argument("--dmax", type=float, default=D_CAL_MAX,
                    help="reject near candidates farther than this "
                         "(default: farthest calibration distance)")
    ap.add_argument("--debug-polar", action="store_true")
    ap.add_argument("--no-gui", action="store_true",
                    help="skip window display (batch test)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--csv", type=str, default=None,
                    help="write per-frame results to this CSV file")
    ap.add_argument("--out", type=str, default=None,
                    help="optional:  record the annotated video (camera "
                         "view + polar radar screen) --  for screencasts "
                         "and tests only,  off by default")
    args = ap.parse_args()

    model, doc, maps = ls.load_calibration(args.cal)

    if args.build_calib:
        cmd_build_calib(maps, args.rcal)
        return

    # vanishing center:  from file if present, else computed
    if os.path.exists(args.rcal):
        with open(args.rcal) as f:
            C = np.array(json.load(f)["center"])
    else:
        C = vanishing_center(maps)
        print(f"vanishing center C = ({C[0]:.1f}, {C[1]:.1f})")

    fine = {T: FineLine(T, maps[T]) for T in FAMILIES}
    rmat = RadarMatrix(fine, C)
    print(f"radar matrix: {rmat.n_d} distance rows x {rmat.n_th} angle "
          f"cols  (0.3 m .. {D_MAX:.0f} m,  {THETA_STEP} deg/col)")

    if args.test:
        junk_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "radar_junk.npy")
        junk = load_junk_mask(junk_path)
        if junk is None:
            junk = build_junk_mask(fine, rmat, args.cal_dir,
                                   out_path=junk_path)
        cmd_test(fine, rmat, args.cal_dir, junk, args.ascii)
        return

    junk_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "radar_junk.npy")
    junk = load_junk_mask(junk_path)
    if junk is None:
        junk = build_junk_mask(fine, rmat, args.cal_dir, out_path=junk_path)

    # ---------------- live / video loop ----------------
    tracker = NearTracker(1 if args.no_persist else PERSIST)
    if args.video:
        src = cv2.VideoCapture(args.video)
        rate = None
    else:
        src = open_camera(args.cam, args.width, args.height)
        rate = probe_camera_rate(src, 2.0)
        src.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"camera delivers {rate:.1f} fps (requested 30) -- "
              f"end-to-end rate is limited by the camera hardware")

    tf = TemporalFilter(3)
    out_w = None
    if args.out:
        # camera:  the container FPS property is a lie (30) --  trust the
        # measured delivery rate so the recording plays at real speed
        if rate is not None:
            ofps = float(rate)
        else:
            ofps = src.get(cv2.CAP_PROP_FPS)
            if not (1.0 <= float(ofps or 0) <= 120.0):
                ofps = 25.0
#        out_w = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
#                                float(ofps), (REF_W + 360, REF_H))
        print(f"writing annotated video {args.out} at {ofps:.2f} fps")

    n = 0
    fps = 0.0
    t_prev = None
    wall_ema = None
    wall_miss = 0
    # live-scene junk mask:  static room features light the SAME (theta,d)
    # cells in every frame, unlike the laser trace which moves with the
    # wall.  Over the first --boot-frames frames count per-cell activity
    # and mask the static ones;  the wall-trace band is excluded (the
    # trace sits at the same rows in every boot frame but it is the
    # measurement itself, not static scene).
    live_counts = (np.zeros((rmat.n_d, rmat.n_th), np.int32)
                   if not args.no_live_mask else None)
    live_walls = set()
    live_mask = None
    csv = open(args.csv, "w") if args.csv else None
    if csv:
        csv.write("frame,wall_m,near_m,near_theta,fam,reported,susp,cpu_ms\n")
    try:
        while True:
            ok, frame = src.read()
            if not ok:
                break
            t0 = time.time()
            if frame.shape[1] != REF_W or frame.shape[0] != REF_H:
                frame = cv2.resize(frame, (REF_W, REF_H))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            junk_now = junk
            if live_mask is not None:
                junk_now = junk | live_mask if junk is not None else live_mask
            res = radar_frame(fine, rmat, gray, args.min_brightness,
                              args.min_row, junk=junk_now,
                              allow_sub_cal=args.sub_cal,
                              d_max=args.dmax, hist=tf)
            if live_counts is not None:
                # 3x3 dilation:  handheld-camera shake smears a static
                # feature over a few cells;  count "lit nearby"
                near = cv2.dilate((res["M"] > 0).astype(np.uint8),
                                  np.ones((3, 3), np.uint8))
                live_counts += (near > 0).astype(np.int32)
                if res["d_wall"] is not None:
                    live_walls.add(round(res["d_wall"], 1))
                if n + 1 >= args.boot_frames and live_mask is None:
                    live_mask = live_counts >= args.live_fraction * (n + 1)
                    for dw in live_walls:
                        live_mask[np.abs(rmat.d - dw) <= 0.8] = False
                    print(f"live junk mask ready:  "
                          f"{int(live_mask.sum())} cells masked "
                          f"(static scene in {n + 1} boot frames, "
                          f"walls {sorted(live_walls)})")
                    tracker.reset()   # static features stay dead;  a real
                                      # new object re-triggers in 3 frames
                    tf.reset()
            dt = time.time() - t0
            reported = tracker.update(res["d_near"], res["theta_near"])
            # wall smoothing:  the room may contain several surfaces at
            # different depths;  an EMA kills the per-frame jumps
            if res["d_wall"] is not None:
                wall_ema = res["d_wall"] if wall_ema is None else \
                    0.7 * wall_ema + 0.3 * res["d_wall"]
                wall_miss = 0
            else:
                wall_miss += 1
                if wall_miss > 30:
                    wall_ema = None
            disp = dict(res)
            disp["d_wall"] = wall_ema
            if t_prev is not None:
                fps = 0.9 * fps + 0.1 / max(time.time() - t_prev, 1e-6)
            t_prev = time.time()
            if (not args.no_gui) or out_w is not None:
                out = annotate(frame, disp, rmat, reported, fps, dt * 1000)
                # the polar heat-map is ALWAYS inline on the right:
                # rows = distance,  cols = angle,  purple = suspicious
                polar = polar_debug_view(res["M"], rmat,
                                         res.get("susp_col"),
                                         acc=res.get("acc"))
                pad = np.zeros((REF_H, 360, 3), np.uint8)
                y0 = (REF_H - 260) // 2
                pad[y0:y0 + 260] = polar
                out = np.hstack([out, pad])
                if out_w is not None:
                    out_w.write(out)
                if not args.no_gui:
                    if args.debug_polar:
                        cv2.imshow("radar screen (rows=dist, cols=angle)",
                                   polar)
                    cv2.imshow("laser radar", out)
            w = res["d_wall"]
            nn = res["d_near"]
            if csv:
                susp_txt = ";".join(f"{a:.0f}-{b:.0f}"
                                    for a, b in res.get("susp", []))
                csv.write(f"{n},{'' if w is None else '%.3f' % w},"
                          f"{'' if nn is None else '%.3f' % nn},"
                          f"{'' if nn is None else '%.1f' % (res['theta_near'] or 0)},"
                          f"{res['fam_near'] or '-'},{int(reported)},"
                          f"{susp_txt},{dt * 1000:.1f}\n")
            wtxt = "%6.2f" % w if w is not None else "    -- "
            if nn is not None:
                ntxt = "%6.2f th%3.0f %s %s" % (nn, res["theta_near"] or 0,
                                               res["fam_near"] or "-",
                                               "REP" if reported else "---")
            else:
                ntxt = "    --  --  ---"
            sl = res.get("susp", [])
            if sl:
                susp_txt = " ".join(f"{a:.0f}-{b:.0f}" for a, b in sl[:3])
                if len(sl) > 3:
                    susp_txt += f" +{len(sl) - 3}"
            else:
                susp_txt = "-"
            line = (f"frame {n:5d}  wall={wtxt}  near={ntxt}   "
                    f"cells={res['n_cells']:3d} streak={res['streak']:2d}  "
                    f"susp={susp_txt:<14} "
                    f"cpu {dt * 1000:5.1f} ms  {fps:5.1f} fps")
            if live_counts is not None and live_mask is None:
                line += f"  [junk boot {n + 1}/{args.boot_frames}]"
            if n % 15 == 0 or reported:
                print(line)
            k = cv2.waitKey(1) & 0xFF if not args.no_gui else 0
            if k in (ord("q"), 27):
                break
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        if csv:
            csv.close()
        if out_w is not None:
            out_w.release()
        cv2.destroyAllWindows()
        src.release()
    print(f"done, {n} frames")


if __name__ == "__main__":
    main()
