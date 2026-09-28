"""
laser_sensor.py
===============
Laser-line distance sensor: camera + 3 laser fans (2 "side" fans producing the
two near-vertical image lines, 1 "bottom" fan producing the horizontal image
line).  What the camera sees are the intersections of the laser fans with the
first surface in front of the camera (calibration board / obstacle).

Those three image lines carry the distance via parallax: each fan is a fixed
3D plane, its trace on a plane at distance d is a 3D line whose projection
moves in the image as d changes.  The two intersection points of the lines
(the corners of the fan-trace rectangle) are the outer boundary points of the
swept volume.

Modules
-------
- Line extraction from a raw image (threshold + iterative least-squares fit
  with outlier rejection, sub-pixel line parameters, brightness profile
  along the line, visible segment extents).
- 3D model: camera pose (height, pitch, yaw, roll, focal length, principal
  point) + 3 fan planes.  Fitted by Levenberg-Marquardt to a set of
  (image, known-distance) calibration samples.
- Measurement: detect the 3 lines in a new image, recover each fan's trace
  distance, report the closest-object distance and the two boundary corner
  points in image and 3D coordinates.

The module is intentionally dependency-light: numpy + opencv + scipy (lmfit
not required, a small LM implementation is included).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# basic 3D helpers
# ----------------------------------------------------------------------------

def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Camera:
    """Pinhole camera.

    World frame: X right, Y up, Z forward (direction of travel).
    Camera sits at (0, height, 0) and is oriented with yaw, pitch (down
    positive), roll (clockwise seen from behind, i.e. positive roll turns the
    image content clockwise).
    """

    def __init__(self, height: float, yaw: float, pitch: float, roll: float,
                 fx: float, fy: float, cx: float, cy: float):
        self.height = height          # camera height above the reference plane
        self.yaw = yaw                # rad
        self.pitch = pitch            # rad, positive = looking down
        self.roll = roll              # rad, positive = clockwise in the image
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    @property
    def R(self) -> np.ndarray:
        # world -> camera: undo orientation.  Camera x right, y down, z fwd.
        R = rot_x(self.pitch) @ rot_y(self.yaw) @ rot_z(self.roll)
        # flip y for the y-down camera frame
        return R @ np.diag([1.0, -1.0, 1.0])

    @property
    def C(self) -> np.ndarray:
        return np.array([0.0, self.height, 0.0])

    def project(self, pts_w: np.ndarray) -> np.ndarray:
        """pts_w: (N,3) world -> (N,2) pixel (nan where behind camera)."""
        pts_w = np.asarray(pts_w, dtype=float)
        pts_c = (pts_w - self.C) @ self.R.T
        z = pts_c[:, 2]
        out = np.full((len(pts_w), 2), np.nan)
        ok = z > 1e-6
        out[ok, 0] = self.fx * pts_c[ok, 0] / z[ok] + self.cx
        out[ok, 1] = self.fy * pts_c[ok, 1] / z[ok] + self.cy
        return out

    def image_line_from_world_line(self, origin_w: np.ndarray,
                                   direction_w: np.ndarray) -> Optional[np.ndarray]:
        """Project a 3D line (origin + direction) to the image plane.

        Returns (a, b, c) with a*x + b*y + c = 0, ||(a,b)|| = 1, or None when
        the line is (nearly) parallel to the image plane.
        """
        R = self.R
        o_c = (np.asarray(origin_w) - self.C) @ R.T
        d_c = np.asarray(direction_w) @ R.T
        # points: o_c + t d_c ; z = z_o + t d_z
        zo, dzo = o_c[2], d_c[2]
        if abs(dzo) < 1e-9:
            # 3D line parallel to image plane: image line is a straight line
            # through the projections of two far samples
            p1 = o_c + 1.0 * d_c
            p2 = o_c - 1.0 * d_c
            if abs(p1[2]) < 1e-9 or abs(p2[2]) < 1e-9:
                return None
            u1 = (self.fx * p1[0] / p1[2] + self.cx, self.fy * p1[1] / p1[2] + self.cy)
            u2 = (self.fx * p2[0] / p2[2] + self.cx, self.fy * p2[1] / p2[2] + self.cy)
        else:
            # vanishing point t->inf and a finite sample
            vp = (self.fx * d_c[0] / dzo + self.cx, self.fy * d_c[1] / dzo + self.cy)
            t0 = 1.0
            p0 = o_c + t0 * d_c
            if abs(p0[2]) < 1e-9:
                return None
            u0 = (self.fx * p0[0] / p0[2] + self.cx, self.fy * p0[1] / p0[2] + self.cy)
            u1, u2 = vp, u0
        a = u1[1] - u2[1]
        b = u2[0] - u1[0]
        c = -(a * u1[0] + b * u1[1])
        n = math.hypot(a, b)
        if n < 1e-9:
            return None
        return np.array([a / n, b / n, c / n])


# ----------------------------------------------------------------------------
# image line extraction
# ----------------------------------------------------------------------------

@dataclass
class ImageLine:
    """A straight line in the image: a*x + b*y + c = 0, ||(a,b)||=1."""
    a: float = 0.0
    b: float = 1.0     # default: horizontal (y = 0)
    c: float = 0.0
    n_points: int = 0
    rms: float = 0.0
    # visible segment: parameter t along (a,b) direction, t_start..t_stop
    t_start: float = -1e9
    t_stop: float = 1e9
    # mean brightness of the supporting pixels
    brightness: float = 0.0
    ok: bool = False

    def point(self, u: float) -> Tuple[float, float]:
        """y at given x for a (near)horizontal line, or x at given y for a
        (near)vertical one.  Always returns (x, y) of the point on the line
        closest to the given coordinate along the line's dominant axis."""
        if abs(self.a) > abs(self.b):      # vertical-ish: a*x + b*y + c = 0 -> x(y)
            return (self._x_at_y(u), u)
        return (u, self._y_at_x(u))

    def _y_at_x(self, x: float) -> float:
        if abs(self.b) < 1e-12:
            return np.nan
        return -(self.a * x + self.c) / self.b

    def _x_at_y(self, y: float) -> float:
        if abs(self.a) < 1e-12:
            return np.nan
        return -(self.b * y + self.c) / self.a

    def distance(self, x: float, y: float) -> float:
        return self.a * x + self.b * y + self.c

    @property
    def is_vertical(self) -> bool:
        return abs(self.a) > abs(self.b)


def _fit_line_to_points(xs: np.ndarray, ys: np.ndarray) -> ImageLine:
    """Total-least-squares line through (xs, ys)."""
    xm, ym = xs.mean(), ys.mean()
    cov = np.cov(np.vstack([xs - xm, ys - ym]))
    evals, evecs = np.linalg.eigh(cov)
    # principal direction = eigenvector of largest eigenvalue
    v = evecs[:, np.argmax(evals)]
    a, b = -v[1], v[0]
    c = -(a * xm + b * ym)
    n = math.hypot(a, b)
    a, b, c = a / n, b / n, c / n
    return ImageLine(a=a, b=b, c=c, n_points=int(len(xs)), ok=True)


def extract_line(img_gray: np.ndarray, mask: np.ndarray, seed: np.ndarray,
                 max_dist: float = 6.0, iters: int = 5) -> ImageLine:
    """Iteratively refine a line given a boolean seed mask.

    seed: boolean mask of candidate bright pixels (e.g. inside a search band).
    Returns an ImageLine fitted to the inliers, with segment extents and
    mean brightness.
    """
    H, W = mask.shape
    sel = np.where(seed)
    if len(sel[0]) < 20:
        return ImageLine(ok=False)
    xs, ys = sel[1].astype(float), sel[0].astype(float)
    line = _fit_line_to_points(xs, ys)
    for _ in range(iters):
        d = line.a * xs + line.b * ys + line.c
        keep = np.abs(d) < max_dist
        if keep.sum() < 20:
            break
        line = _fit_line_to_points(xs[keep], ys[keep])
    d = line.a * xs + line.b * ys + line.c
    keep = np.abs(d) < max_dist
    if keep.sum() < 20:
        line.ok = False
        return line
    kx, ky = xs[keep], ys[keep]
    line.n_points = int(keep.sum())
    line.rms = float(np.sqrt(np.mean(d[keep] ** 2)))
    line.brightness = float(img_gray[ky.astype(int), kx.astype(int)].mean())
    # segment extents along the line direction
    dirv = np.array([line.b, -line.a])          # unit direction on the line
    t = (kx - 0) * dirv[0] + (ky - 0) * dirv[1]
    line.t_start, line.t_stop = float(t.min()), float(t.max())
    line.ok = True
    return line


def extract_line_guided(img: np.ndarray, model: Model, fan_name: str,
                        d_guess: float, strip: float = 60.0,
                        max_err: float = 30.0, ctx: Optional[tuple] = None
                        ) -> Tuple[ImageLine, float]:
    """Fit the image line of one fan using the model prediction at d_guess
    as the search window.

    Works on the sparse set of bright pixels (not a full-image grid), which
    keeps each call in a few milliseconds so a grid of d_guess values is
    cheap to sweep.  `ctx` = (gray, thr, pxs, pys, H, W) may be passed in
    pre-computed by guided_context() to avoid redoing the work per call.

    Returns (line, err) where err is the max deviation (px) of the fitted
    line from the predicted one; err > max_err means the fit should be
    discarded (e.g. the window grabbed the wrong structure).
    """
    if ctx is not None:
        g, thr, pxs, pys, H, W = ctx
    else:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        vals = np.sort(g.ravel())
        thr = max(25.0, 0.5 * (float(vals[int(0.30 * len(vals))]) + float(vals[int(0.80 * len(vals))])))
        mask = g > thr
        H, W = g.shape
        pys, pxs = np.where(mask)
        if len(pxs) == 0:
            return ImageLine(ok=False), float('inf')
        k = ((pxs > 0.25 * W) & (pxs < 0.75 * W) &
             (pys > 0.15 * H) & (pys < 0.85 * H))
        pxs, pys = pxs[k], pys[k]
    if len(pxs) == 0:
        return ImageLine(ok=False), float('inf')
    pred = dict(model.predict_lines(d_guess))[fan_name]
    if pred is None:
        return ImageLine(ok=False), float('inf')
    a, b, c = pred
    sel = np.abs(a * pxs + b * pys + c) < strip
    if int(sel.sum()) < 30:
        return ImageLine(ok=False), float('inf')
    seed = np.zeros((H, W), np.uint8)
    seed[pys[sel], pxs[sel]] = 1
    line = extract_line(g, seed, seed)
    if not line.ok:
        return line, float('inf')
    # reject fits that only span a tiny region (noise clusters): a real
    # laser line in the central window spans hundreds of pixels
    dd = np.array([line.b, -line.a])
    tv = pxs[sel] * dd[0] + pys[sel] * dd[1]
    if (tv.max() - tv.min()) < 120.0:
        line.ok = False
        return line, float('inf')
    # error between fitted and predicted lines, measured at two points on
    # the fitted line (near the image center)
    cx, cy = W / 2.0, H / 2.0
    s = line.a * cx + line.b * cy + line.c
    fx, fy = cx - line.a * s, cy - line.b * s        # foot of perpendicular
    dx, dy = line.b, -line.a                          # unit direction of fitted line
    err = 0.0
    for t in (-100.0, 0.0, 100.0):
        px, py = fx + t * dx, fy + t * dy
        err = max(err, abs(a * px + b * py + c))
    if err > max_err:
        line.ok = False
    return line, float(err)


def guided_context(img: np.ndarray) -> tuple:
    """Pre-compute the per-frame context (gray, thr, bright points in the
    central window, image size) used by extract_line_guided, so that a
    whole sweep of d_guess values costs one sort + one where."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    vals = np.sort(g.ravel())
    thr = max(25.0, 0.5 * (float(vals[int(0.30 * len(vals))]) + float(vals[int(0.80 * len(vals))])))
    mask = g > thr
    H, W = g.shape
    pys, pxs = np.where(mask)
    k = ((pxs > 0.25 * W) & (pxs < 0.75 * W) &
         (pys > 0.15 * H) & (pys < 0.85 * H))
    return (g, thr, pxs[k], pys[k], H, W)


def _band_mask_from_rowprofile(mask: np.ndarray, peak_row: int, frac: float = 0.5):
    H = mask.shape[0]
    rowcnt = mask.sum(axis=1)
    peak = rowcnt[peak_row]
    lo = hi = peak_row
    while lo > 0 and rowcnt[lo - 1] > frac * peak:
        lo -= 1
    while hi < H - 1 and rowcnt[hi + 1] > frac * peak:
        hi += 1
    band = np.zeros_like(mask)
    band[max(0, lo - 1):hi + 2] = mask[max(0, lo - 1):hi + 2]
    return band


def _cluster_columns(colcnt: np.ndarray, thresh: float, gap: int = 40):
    cols = np.where(colcnt > thresh)[0]
    clusters = []
    for c in cols:
        if clusters and c - clusters[-1][-1] <= gap:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    return clusters


def _v_bands(mask: np.ndarray, min_count: int = 100, min_height: int = 150,
             max_width: int = 60, center_frac=(0.25, 0.75)):
    """Candidate bands for the (near-)vertical laser lines.

    Uses an absolute pixel-count threshold (robust to bright non-laser blobs)
    plus shape filters: thin in x, tall in y, and located in the central part
    of the frame (the laser lines stay inside ~the central half; bright edge
    artifacts/room features outside it are ignored).
    """
    colcnt = mask.sum(axis=0)
    W = mask.shape[1]
    clusters = _cluster_columns(colcnt, min_count)
    bands = []
    for cl in clusters:
        c0, c1 = cl[0], cl[-1]
        if (c1 - c0) > max_width:          # wide: blob, not a line
            continue
        cx = (c0 + c1) / 2.0
        if not (center_frac[0] * W <= cx <= center_frac[1] * W):
            continue
        sub = mask[:, c0:c1 + 1]
        rows = np.where(sub.any(axis=1))[0]
        if len(rows) == 0:
            continue
        h = rows.max() - rows.min()
        if h < min_height:                 # not tall enough to be a laser line
            continue
        # thin in x on average
        band = np.zeros_like(mask)
        band[:, max(0, c0 - 4):c1 + 5] = mask[:, max(0, c0 - 4):c1 + 5]
        bands.append((band, cx))
    bands.sort(key=lambda t: t[1])
    return bands


def _h_band(mask: np.ndarray, min_count: int = 100, min_width: int = 300):
    """Candidate band for the (near-)horizontal laser line (or None)."""
    rowcnt = mask.sum(axis=0 if False else 1)
    if rowcnt.max() < min_count:
        return None
    peak = int(np.argmax(rowcnt))
    band = _band_mask_from_rowprofile(mask, peak)
    cols = np.where(band.any(axis=0))[0]
    if len(cols) == 0 or (cols.max() - cols.min()) < min_width:
        return None
    return band


def detect_lines(img: np.ndarray, thr: Optional[float] = None,
                 min_h_count: int = 100, min_v_count: int = 100) -> dict:
    """Detect the three laser lines in one image.

    Returns dict with keys 'H', 'V1' (left), 'V2' (right); each an ImageLine.
    Missing / unreliable lines have .ok == False.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = g.shape
    if thr is None:
        # adaptive threshold: midpoint of 30th and 80th percentile, floored
        vals = np.sort(g.ravel())
        thr = max(25.0, 0.5 * (float(vals[int(0.30 * len(vals))]) + float(vals[int(0.80 * len(vals))])))
    mask = g > thr
    out = {"H": ImageLine(ok=False), "V1": ImageLine(ok=False), "V2": ImageLine(ok=False)}

    # ---- horizontal line -------------------------------------------------
    band = _h_band(mask, min_h_count)
    if band is not None:
        line = extract_line(g, mask, band)
        if line.ok and abs(line.b) > abs(line.a):      # must be near-horizontal
            out["H"] = line

    # ---- vertical lines ----------------------------------------------------
    bands = _v_bands(mask, min_v_count)
    cands = []
    for band, cx in bands:
        line = extract_line(g, mask, band)
        if line.ok and abs(line.a) > abs(line.b):      # must be near-vertical
            cands.append((line, cx))
    if len(cands) >= 1:
        out["V1"] = cands[0][0]
    if len(cands) >= 2:
        out["V2"] = cands[-1][0]
    return out


def line_intersection(l1: ImageLine, l2: ImageLine) -> Optional[Tuple[float, float]]:
    A = np.array([[l1.a, l1.b], [l2.a, l2.b]])
    b = np.array([-l1.c, -l2.c])
    try:
        x, y = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return float(x), float(y)


def fan_pair_geometry(cam: "Camera", fan1: "Fan", fan2: "Fan"):
    """Image geometry of the intersection point of two laser planes.

    The 3-D point P(d) = fan1 ∩ fan2 ∩ {z = d} moves along a fixed 3-D line
    as d varies; its image projection is therefore a straight line, and its
    limit for d -> infinity is a finite vanishing point.  This is what makes
    the corner points 'slide' along straight image lines that all converge
    at that point.

    Returns ((vp_x, vp_y), (a, b, c)) with the image line a*x + b*y + c = 0
    (|n|=1), or None when degenerate.
    """
    n1, c1 = np.asarray(fan1.normal, float), fan1.c
    n2, c2 = np.asarray(fan2.normal, float), fan2.c
    if np.linalg.norm(np.cross(n1, n2)) < 1e-9:
        return None
    A = np.array([n1, n2, [0.0, 0.0, 1.0]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    # direction of P(d) per unit increase of z (= A^-1 [0,0,1])
    w = np.linalg.solve(A, np.array([0.0, 0.0, 1.0]))
    d_c = w @ cam.R.T
    if abs(d_c[2]) < 1e-9:
        return None
    vp = (cam.fx * d_c[0] / d_c[2] + cam.cx,
          cam.fy * d_c[1] / d_c[2] + cam.cy)
    # project sample points of the 3-D line to get the image line
    pts = [vp]
    for d in (1.0, 3.0, 8.0, 25.0):
        p = np.linalg.solve(A, np.array([c1, c2, d]))
        pc = (p - cam.C) @ cam.R.T
        if pc[2] <= 1e-6:
            continue
        pts.append((cam.fx * pc[0] / pc[2] + cam.cx,
                    cam.fy * pc[1] / pc[2] + cam.cy))
    if len(pts) < 2:
        return None
    pts = np.asarray(pts)
    xs, ys = pts[:, 0], pts[:, 1]
    xm, ym = float(xs.mean()), float(ys.mean())
    cov = np.cov(np.vstack([xs - xm, ys - ym]))
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))]
    a, b = -v[1], v[0]
    nrm = math.hypot(a, b)
    if nrm < 1e-9:
        return None
    a, b = a / nrm, b / nrm
    c = -(a * xm + b * ym)
    return (float(vp[0]), float(vp[1])), (a, b, c)


def corner_trajectories(maps: dict) -> dict:
    """Empirical image trajectories of the laser-line corner points.

    For the pairs (L,B) and (R,B) the corner = intersection of the two
    fans' image lines.  As the target distance d grows, these corners
    move along (nearly) straight image lines and converge to finite
    'vanishing' points.  We fit both:

      * a straight line through the 28 measured corner positions
        (the path the corner slides along in the image), and
      * p(d) = A + B/d per coordinate, whose intercept A is the image
        point the corner converges to as d -> infinity.

    Returns {"LxB": {"vp": (x,y), "line": (a,b,c)}, "RxB": {...}} using
    the calibrated LineMaps in `maps` (keys "L", "R", "B").
    """
    out = {}
    for key, (fa, fb) in (("LxB", ("L", "B")), ("RxB", ("R", "B"))):
        if fa not in maps or fb not in maps:
            continue
        ds, xs, ys = [], [], []
        for d in maps[fa].ds:
            p = line_intersection(maps[fa].predict(d), maps[fb].predict(d))
            if p is None:
                continue
            ds.append(d); xs.append(p[0]); ys.append(p[1])
        if len(ds) < 3:
            continue
        ds = np.array(ds); xs = np.array(xs); ys = np.array(ys)
        t = 1.0 / ds

        def ols(y):
            M = np.vstack([np.ones_like(t), t]).T
            coef, *_ = np.linalg.lstsq(M, y, rcond=None)
            return coef

        cx = ols(xs); cy = ols(ys)
        vp = (float(cx[0]), float(cy[0]))
        xm, ym = float(xs.mean()), float(ys.mean())
        cov = np.cov(np.vstack([xs - xm, ys - ym]))
        evals, evecs = np.linalg.eigh(cov)
        v = evecs[:, int(np.argmax(evals))]
        a, b = -v[1], v[0]
        nrm = math.hypot(a, b)
        if nrm < 1e-9:
            continue
        a, b = a / nrm, b / nrm
        c = -(a * xm + b * ym)
        out[key] = {"vp": vp, "line": (float(a), float(b), float(c))}
    return out


# ----------------------------------------------------------------------------
# line-segment detection (for obstacles: one laser line may kink into several
# straight sub-segments at different distances)
# ----------------------------------------------------------------------------

@dataclass
class LineSegment:
    """A straight sub-segment of a laser line: the part reflecting from one
    single surface at one distance."""
    line: ImageLine
    t_start: float            # extent along the line direction
    t_stop: float
    brightness: float         # mean brightness of its pixels
    n_points: int

    def midpoint(self) -> Tuple[float, float]:
        l = self.line
        d = np.array([l.b, -l.a])
        # point on the line closest to the image center-ish origin:
        # line: a x + b y + c = 0  -> foot of perpendicular from (0,0):
        x0 = -l.a * l.c
        y0 = -l.b * l.c
        t = 0.5 * (self.t_start + self.t_stop)
        return (float(x0 + t * d[0]), float(y0 + t * d[1]))

    def t_of_point(self, p: Tuple[float, float]) -> float:
        l = self.line
        d = np.array([l.b, -l.a])
        return float(p[0] * d[0] + p[1] * d[1])

    def contains_point(self, p: Tuple[float, float], margin: float = 15.0) -> bool:
        t = self.t_of_point(p)
        return self.t_start - margin <= t <= self.t_stop + margin


def _residuals(l: ImageLine, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return l.a * xs + l.b * ys + l.c


def split_segments(img_gray: np.ndarray, mask: np.ndarray, band: np.ndarray,
                   max_dist: float = 6.0, max_splits: int = 3,
                   min_split_resid: float = 3.0) -> List[LineSegment]:
    """Detect straight sub-segments inside a laser-line band.

    Starts with a global fit; where the band deviates systematically
    (residual beyond min_split_resid on both sides) it is split recursively.
    """
    ys0, xs0 = np.where(band)
    segs: List[Tuple[np.ndarray, np.ndarray]] = [(xs0, ys0)]
    for _ in range(max_splits):
        progressed = False
        new_segs = []
        for (xs, ys) in segs:
            if len(xs) < 40:
                new_segs.append((xs, ys))
                continue
            l = _fit_line_to_points(xs, ys)
            r = _residuals(l, xs, ys)
            # order points along the line: by y for vertical lines, by x
            # otherwise (the slow-varying coordinate)
            axis = ys if abs(l.a) > abs(l.b) else xs
            so = np.argsort(axis, kind="stable")
            xo, yo = xs[so], ys[so]
            ro = r[so]
            split_i = None
            for i in range(1, len(xo) - 1):
                if (ro[i - 1] > min_split_resid and ro[i + 1] < -min_split_resid) or \
                   (ro[i - 1] < -min_split_resid and ro[i + 1] > min_split_resid):
                    split_i = i
                    break
            if split_i is None:
                # maybe a global bend: fit two lines with a 1-D split search
                best = None
                n_pts = len(xo)
                for i in range(20, n_pts - 20, max(1, n_pts // 20)):
                    a_part = (xo[:i], yo[:i])
                    b_part = (xo[i:], yo[i:])
                    la = _fit_line_to_points(*a_part)
                    lb = _fit_line_to_points(*b_part)
                    ra = _residuals(la, *a_part)
                    rb = _residuals(lb, *b_part)
                    both_rms2 = float(np.mean(np.concatenate([ra ** 2, rb ** 2])))
                    if np.all(np.abs(ra) < max_dist) and np.all(np.abs(rb) < max_dist) \
                            and both_rms2 < 0.25 * float(np.mean(r ** 2)):
                        cand = (both_rms2, i)
                        if best is None or cand[0] < best[0]:
                            best = cand
                if best is not None:
                    i = best[1]
                    new_segs.append((xo[:i], yo[:i]))
                    new_segs.append((xo[i:], yo[i:]))
                    progressed = True
                    continue
                new_segs.append((xs, ys))
                continue
            # hard sign-flip split at i
            new_segs.append((xo[:split_i + 1], yo[:split_i + 1]))
            new_segs.append((xo[split_i + 1:], yo[split_i + 1:]))
            progressed = True
        if not progressed:
            break
        segs = new_segs

    out: List[LineSegment] = []
    for (xs, ys) in segs:
        if len(xs) < 20:
            continue
        l = _fit_line_to_points(xs, ys)
        d = _residuals(l, xs, ys)
        keep = np.abs(d) < max_dist
        if keep.sum() < 15:
            continue
        xs, ys = xs[keep], ys[keep]
        l = _fit_line_to_points(xs, ys)
        d = np.array([l.b, -l.a])
        t = xs * d[0] + ys * d[1]
        segs_l = LineSegment(line=l, t_start=float(t.min()), t_stop=float(t.max()),
                             brightness=float(img_gray[ys.astype(int), xs.astype(int)].mean()),
                             n_points=int(len(xs)))
        out.append(segs_l)
    out.sort(key=lambda s: s.t_start)
    return out


# ----------------------------------------------------------------------------
# 1-D per-laser calibration maps
# ----------------------------------------------------------------------------

def _moving_median(vals: List[float], window: int = 1) -> List[float]:
    n = len(vals)
    if n < 3:
        return list(vals)
    ext = [vals[0]] * window + vals + [vals[-1]] * window
    out = []
    for i in range(n):
        out.append(float(np.median(ext[i:i + 2 * window + 1])))
    return out


def _isotonic_regression(y: List[float]) -> List[float]:
    """Non-decreasing isotonic regression (pool adjacent violators)."""
    n = len(y)
    blocks = [[i, i, float(v)] for i, v in enumerate(y)]  # [start, end, sum]
    for _ in range(n):
        i = 0
        while i < len(blocks) - 1:
            b1, b2 = blocks[i], blocks[i + 1]
            m1 = b1[2] / (b1[1] - b1[0] + 1)
            m2 = b2[2] / (b2[1] - b2[0] + 1)
            if m1 > m2:
                b1[1] = b2[1]
                b1[2] += b2[2]
                blocks.pop(i + 1)
            else:
                i += 1
    out = [0.0] * n
    for s, e, ssum in blocks:
        m = ssum / (e - s + 1)
        for k in range(s, e + 1):
            out[k] = m
    return out


class LineMap:
    """Empirical 1-D calibration map for one laser line.

    Maps distance <-> image line.  For each calibration distance d_i the
    line measured in the calibration set is stored in a unique 2-parameter
    form (no sign ambiguity):
        vertical line (L/R):  x = m*y + x0   (x0 = x at y=360)
        horizontal line (B):  y = m*x + y0   (y0 = y at x=640)
    Distances between calibration points are interpolated piecewise linearly.
    This captures whatever per-line behaviour the rigid plane model cannot
    (beam footprint changes, small geometric non-idealities).
    """

    def __init__(self, fan_name: str, vertical: bool, direction: int = +1):
        self.fan_name = fan_name
        self.vertical = vertical
        self.direction = direction          # sign of dx0/dd: +1 L, -1 R/B
        self.ds: List[float] = []
        self.x0s: List[float] = []
        self.ms: List[float] = []

    @classmethod
    def build(cls, fan_name: str, meas_list: List[dict]) -> "LineMap":
        direction = {"L": +1, "R": -1, "B": -1}.get(fan_name, +1)
        m = cls(fan_name, fan_name != "B", direction)
        for rec in meas_list:
            line = rec[fan_name]
            if not line.ok:
                continue
            a, b, c = line.a, line.b, line.c
            if m.vertical:
                if abs(a) < 1e-9:
                    continue
                mv = -(b / a)
                x0 = -(b * 360.0 + c) / a
            else:
                if abs(b) < 1e-9:
                    continue
                mv = -(a / b)
                x0 = -(a * 640.0 + c) / b
            m.ds.append(float(rec["d"]))
            m.x0s.append(float(x0))
            m.ms.append(float(mv))
        if m.ds:
            order = np.argsort(m.ds)
            m.ds = [m.ds[i] for i in order]
            m.x0s = [m.x0s[i] for i in order]
            m.ms = [m.ms[i] for i in order]
            # R is not a rigid plane and has genuine regime jumps (~9 m and
            # ~13 m): keep both x0 and m raw (smoothing would merge the
            # jumps and create false inversion minima).  L and B are smooth:
            # median + isotonic.
            if m.fan_name != "R":
                m.x0s = _moving_median(m.x0s)
                m.ms = _moving_median(m.ms)
                y = m.x0s if m.direction > 0 else [-v for v in m.x0s]
                fit = _isotonic_regression(y)
                m.x0s = fit if m.direction > 0 else [-v for v in fit]
        return m

    # slope-to-pixel equivalence for the residual metric: a slope error of
    # dv over a ~700 px line spans ~700*dv px at the ends; we use 150 as a
    # compromise weighting for quality scoring.
    SLOPE_W = 150.0

    def _pair_moves(self):
        """Per-pair movement in equivalent pixels between adjacent points."""
        out = []
        for i in range(len(self.ds) - 1):
            dx = abs(self.x0s[i + 1] - self.x0s[i])
            dm = abs(self.ms[i + 1] - self.ms[i]) * self.SLOPE_W
            out.append(float(np.hypot(dx, dm)))
        return out

    def quality(self, d: float) -> float:
        """Local information content at distance d: total movement of the
        map within +/-1.5 m around d (equivalent pixels).  Small values
        mean the laser is saturated there and d is poorly determined."""
        if not self.ds:
            return 0.0
        moves = self._pair_moves()
        tot = 0.0
        for i, mv in enumerate(moves):
            mid = 0.5 * (self.ds[i] + self.ds[i + 1])
            if abs(mid - d) <= 1.5:
                tot += mv
        return tot

    def fit_distance(self, line: ImageLine) -> Optional[Tuple[float, float, float]]:
        """Find d whose predicted line matches `line`.
        Returns (d, max_abs_residual_px, quality) or None."""
        if not line.ok or len(self.ds) < 2:
            return None
        a, b, c = line.a, line.b, line.c
        if self.vertical and abs(a) < 1e-9:
            return None
        if (not self.vertical) and abs(b) < 1e-9:
            return None

        def err(d: float) -> float:
            p = self.predict(d)
            r = line_residual(p, line)
            return float(np.max(np.abs(r)))

        lo, hi = self.ds[0], self.ds[-1]
        # nested grid refinement: coarse scan, then tighten the window.  More
        # robust than ternary search when the err surface has shallow
        # secondary minima (non-plane laser behaviour).
        d_best, e_best = lo, float("inf")
        for n_scan, step in ((300, None), (40, None), (10, None)):
            if n_scan == 300:
                ds_scan = np.linspace(lo, hi, n_scan)
            else:
                span = 0.2 if n_scan == 40 else 0.02
                a = max(lo, d_best - span)
                b = min(hi, d_best + span)
                ds_scan = np.linspace(a, b, n_scan)
            errs = np.array([err(d) for d in ds_scan])
            i = int(np.argmin(errs))
            if errs[i] < e_best:
                e_best = float(errs[i])
                d_best = float(ds_scan[i])
        return d_best, e_best, self.quality(d_best)

    def __len__(self):
        return len(self.ds)

    def range(self):
        return (self.ds[0], self.ds[-1]) if self.ds else None

    def predict(self, d: float) -> ImageLine:
        """Predicted image line for distance d (clamped to the map range)."""
        if not self.ds:
            return ImageLine(ok=False)
        d = min(max(d, self.ds[0]), self.ds[-1])
        x0 = float(np.interp(d, self.ds, self.x0s))
        mv = float(np.interp(d, self.ds, self.ms))
        if self.vertical:
            # x = mv*y + x0  ->  x - mv*y + (mv*360 - x0) = 0
            a, b, c = 1.0, -mv, mv * 360.0 - x0
        else:
            # y = mv*x + y0  ->  -mv*x + y + (mv*640 - y0) = 0
            a, b, c = -mv, 1.0, mv * 640.0 - x0
        n = math.hypot(a, b)
        return ImageLine(a=a / n, b=b / n, c=c / n, ok=True)

    def as_dict(self) -> dict:
        return {"fan_name": self.fan_name, "vertical": self.vertical,
                "direction": self.direction,
                "d": self.ds, "x0": self.x0s, "m": self.ms}

    @classmethod
    def from_dict(cls, d: dict) -> "LineMap":
        m = cls(d["fan_name"], d["vertical"], d.get("direction", +1))
        m.ds = list(d["d"]); m.x0s = list(d["x0"]); m.ms = list(d["m"])
        return m


# ----------------------------------------------------------------------------
# 3D fan model
# ----------------------------------------------------------------------------

@dataclass
class Fan:
    """A laser fan = a plane  n . p = c  (n unit).

    Parameters: normal direction (2 DOF) + plane offset c (1 DOF) = 3 DOF,
    exactly the degrees of freedom of a plane.
    """
    name: str
    normal: np.ndarray          # (3,) unit
    c: float = 0.0              # plane offset: n . p = c

    def trace_on_plane(self, d: float):
        """Intersection line (origin, direction) of this fan with the plane
        Z = d (the target/obstacle plane)."""
        n = self.normal
        # line direction = n x z-axis = (n_y, -n_x, 0)
        dv = np.array([n[1], -n[0], 0.0])
        if np.linalg.norm(dv) < 1e-9:
            return None
        dv = dv / np.linalg.norm(dv)
        # a point on n_x x + n_y y = c at z = d
        if abs(n[0]) > abs(n[1]):
            p0 = np.array([self.c / n[0], 0.0, d])
        else:
            p0 = np.array([0.0, self.c / n[1], d])
        return p0, dv


@dataclass
class Model:
    """Camera + three fans."""
    cam: Camera
    fans: List[Fan] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "camera": {
                "height": self.cam.height, "yaw": self.cam.yaw,
                "pitch": self.cam.pitch, "roll": self.cam.roll,
                "fx": self.cam.fx, "fy": self.cam.fy,
                "cx": self.cam.cx, "cy": self.cam.cy,
            },
            "fans": [
                {"name": f.name, "normal": list(f.normal), "c": f.c}
                for f in self.fans
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        c = d["camera"]
        cam = Camera(c["height"], c["yaw"], c["pitch"], c["roll"],
                     c["fx"], c["fy"], c["cx"], c["cy"])
        fans = [Fan(f["name"], np.asarray(f["normal"], float), f["c"])
                for f in d["fans"]]
        return cls(cam, fans)

    # ---- parameter packing for optimization --------------------------------
    def to_params(self) -> np.ndarray:
        p = [self.cam.height, self.cam.yaw, self.cam.pitch, self.cam.roll,
             self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy]
        for f in self.fans:
            p += [f.normal[0], f.normal[1], f.normal[2], f.c]
        return np.asarray(p, float)

    def set_params(self, p: np.ndarray) -> None:
        self.cam.height = p[0]
        self.cam.yaw = p[1]
        self.cam.pitch = p[2]
        self.cam.roll = p[3]
        self.cam.fx, self.cam.fy = p[4], p[5]
        self.cam.cx, self.cam.cy = p[6], p[7]
        for i, f in enumerate(self.fans):
            o = 8 + i * 4
            n = np.array([p[o], p[o + 1], p[o + 2]])
            nn = np.linalg.norm(n)
            f.normal = n / nn if nn > 1e-9 else np.array([1.0, 0.0, 0.0])
            f.c = p[o + 3]

    def predict_lines(self, d: float) -> List[Tuple[str, Optional[np.ndarray]]]:
        """Predicted image lines (a,b,c) for the three fans on plane Z=d,
        in order [L, R, B]."""
        res = []
        for f in self.fans:
            tr = f.trace_on_plane(d)
            if tr is None:
                res.append((f.name, None))
                continue
            o, dv = tr
            res.append((f.name, self.cam.image_line_from_world_line(o, dv)))
        return res


# ----------------------------------------------------------------------------
# calibration (Levenberg-Marquardt on 2D line residuals)
# ----------------------------------------------------------------------------

def line_residual(r_pred, r_meas) -> np.ndarray:
    """Residual between two normalized lines (a,b,c) or ImageLines.
    Returns (2,): signed distance of the predicted line at two probe points
    relative to the measured line (robust to the (a,b,c) sign flip)."""
    if isinstance(r_pred, ImageLine):
        r_pred = (r_pred.a, r_pred.b, r_pred.c)
    if isinstance(r_meas, ImageLine):
        r_meas = (r_meas.a, r_meas.b, r_meas.c)
    ap, bp, cp = r_pred
    am, bm, cm = r_meas
    # orient predicted like measured
    if ap * am + bp * bm < 0:
        ap, bp, cp = -ap, -bp, -cp
    # probe points: on measured line at x = cx-100, cx+100 style -> use its
    # dominant direction
    dv = np.array([bm, -am])
    # probe around the point of the measured line closest to the image origin
    center = np.array([-am * cm, -bm * cm])
    out = []
    for s in (-1.0, 1.0):
        pt = center + s * 100.0 * dv
        out.append(ap * pt[0] + bp * pt[1] + cp)
    return np.asarray(out)


def lines_from_samples(samples: List[Tuple[np.ndarray, float]]) -> List[dict]:
    """Run plain (unassisted) line detection on each image.
    Returns [{'L': ImageLine, 'R': ImageLine, 'B': ImageLine, 'd': float}, ...]
    """
    out = []
    for img, d in samples:
        L = detect_lines(img)
        out.append({"L": L["V1"], "R": L["V2"], "B": L["H"], "d": d})
    return out


def calibrate(meas_list: List[dict],
              model: Optional[Model] = None,
              max_iter: int = 300, verbose: bool = True) -> Tuple[Model, dict]:
    """Fit the Model to pre-detected lines.

    meas_list: [{'L': ImageLine, 'R': ImageLine, 'B': ImageLine, 'd': float}, ...]
    (see lines_from_samples / extract_line_guided).  Returns (model, report).
    """
    if model is None:
        model = initial_model()
    p0 = model.to_params()
    meas = meas_list

    def residuals(p: np.ndarray) -> np.ndarray:
        model.set_params(p)
        r = []
        for m in meas:
            pred = {name: l for name, l in model.predict_lines(m["d"])}
            for key in ("L", "R", "B"):
                pm, mm = pred[key], m[key]
                if not mm.ok or pm is None:
                    continue
                r.append(line_residual(pm, mm))
        return np.concatenate(r) if r else np.zeros(2)

    # ---- simple Levenberg-Marquardt with numerical Jacobian ----------------
    p = p0.copy()
    f = residuals(p)
    cost = float(f @ f)
    nu = 1.0
    it = 0
    history = []
    while it < max_iter:
        it += 1
        # numerical Jacobian
        J = np.zeros((len(f), len(p)))
        step = 1e-5 * (np.abs(p) + 1e-3)
        for j in range(len(p)):
            pp, pm = p.copy(), p.copy()
            pp[j] += step[j]; pm[j] -= step[j]
            fp, fm = residuals(pp), residuals(pm)
            J[:, j] = (fp - fm) / (2 * step[j])
        H = J.T @ J
        g = J.T @ f
        improved = False
        for _ in range(30):
            try:
                dp = np.linalg.solve(H + nu * np.trace(H) / len(H) * np.eye(len(p)), -g)
            except np.linalg.LinAlgError:
                break
            pn = p + dp
            fn = residuals(pn)
            cn = float(fn @ fn)
            if cn < cost:
                p, f, cost = pn, fn, cn
                nu = max(nu / 3.0, 1e-3)
                improved = True
                break
            nu *= 3
        if not improved:
            break
        if verbose and it % 10 == 0:
            print(f"    iter {it:3d}  cost {cost:.4f}")
        history.append(cost)
        if it > 1 and abs(history[-2] - history[-1]) < 1e-8 * max(1.0, history[-1]):
            break
    model.set_params(p)

    # ---- report -------------------------------------------------------------
    report = {"cost": cost, "per_sample": [], "converged": bool(history and history[-1] <= history[0])}
    for m in meas:
        pred = {name: l for name, l in model.predict_lines(m["d"])}
        errs = []
        for key in ("L", "R", "B"):
            if m[key].ok and pred[key] is not None:
                r = line_residual(pred[key], m[key])
                errs.append(float(np.max(np.abs(r))))
        report["per_sample"].append({"d": m["d"], "max_abs_px": max(errs) if errs else None,
                                     "lines_ok": [m[k].ok for k in ("L", "R", "B")]})
    return model, report


def initial_model() -> Model:
    """A sensible starting point (units: meters, radians, pixels).

    World: X right, Y up, Z forward.  Camera at height h.  Fans:
    L/R are near-vertical planes beside the camera, B a near-horizontal
    plane below it.
    """
    cam = Camera(height=0.25, yaw=0.0, pitch=0.05, roll=0.0,
                 fx=2000.0, fy=2000.0, cx=640.0, cy=360.0)
    # near-vertical left fan: normal mostly +X, slight -Z tilt, slight Y
    nL = np.array([0.995, 0.02, 0.10]); nL /= np.linalg.norm(nL)
    nR = np.array([-0.995, 0.02, 0.10]); nR /= np.linalg.norm(nR)
    nB = np.array([0.0, 0.90, 0.44]);    nB /= np.linalg.norm(nB)
    fans = [
        Fan("L", nL, c=0.05),
        Fan("R", nR, c=-0.05),
        Fan("B", nB, c=0.15),
    ]
    return Model(cam, fans)


# ----------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------

@dataclass
class Measurement:
    distances: dict            # fan name -> distance (m)
    closest: Optional[float]   # min distance among detected lines
    corners: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]  # image pts
    corners_3d: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]
    lines: dict               # fan name -> ImageLine
    details: dict = field(default_factory=dict)
    near_features: list = field(default_factory=list)   # bright corridor features
    near_min: Optional[float] = None        # closest distance implied by a near feature

    def as_dict(self) -> dict:
        dd = asdict(self)
        dd["lines"] = {k: {"a": v.a, "b": v.b, "c": v.c, "n": v.n_points,
                           "brightness": v.brightness, "ok": v.ok}
                       for k, v in self.lines.items()}
        return dd


def image_threshold(gray: np.ndarray) -> float:
    """Additive brightness threshold for laser lines in a dark scene."""
    vals = np.sort(gray.ravel())
    return max(25.0, 0.5 * (float(vals[int(0.30 * len(vals))]) +
                            float(vals[int(0.80 * len(vals))])))


def _point_distance_to_b(b_map, x: float, y: float, max_err: float = 10.0,
                         model=None):
    """Distance (m) of the surface whose B-laser line passes through image
    point (x, y): scan the calibrated B map and take the best match.
    Returns (d, err) or None when no d within max_err px explains the point.
    When the match is at the near calibration limit (or absent), refine with
    a 3D ray/B-fan-plane intersection (continuous, like the segment
    fallback), so sub-range distances below 2 m are still estimated."""
    if b_map is None:
        return None
    best = None
    for d in b_map.ds:
        ln = b_map.predict(d)
        if ln is None or abs(ln.b) < 1e-9:
            continue
        ypred = -(ln.a * x + ln.c) / ln.b
        err = abs(ypred - y)
        if best is None or err < best[1]:
            best = (float(d), float(err))
    if best is not None and best[1] <= max_err:
        d = best[0]
        if d < 2.02:
            # near/below the nearest calibration distance: the map scan is
            # saturated; extrapolate the empirical map, else the 3D plane
            d3 = _extrapolate_b_distance(b_map, x, y)
            if d3 is None:
                d3 = _point_b_plane_distance(model, x, y)
            if d3 is not None:
                return (d3, best[1])
        return best
    return None


def _extrapolate_b_distance(b_map, x: float, y: float,
                            lo: float = 0.4, hi: float = 2.05) -> Optional[float]:
    """Distance for a point below the nearest calibrated B line: extrapolate
    the empirical B map y0(d) = A + B/d (fit on all calibration points) to
    sub-range distances.  Returns None if the estimate leaves [lo, hi]."""
    if b_map is None or len(b_map.ds) < 3:
        return None
    try:
        ds = np.asarray(b_map.ds, float)
        y0s = np.asarray(b_map.x0s, float)     # for B: y at x=640
        t = 1.0 / ds
        A, B = np.linalg.lstsq(
            np.vstack([np.ones_like(t), t]).T, y0s, rcond=None)[0]
        m_avg = float(np.mean(b_map.ms))
        y0_est = y - m_avg * (x - 640.0)
        if y0_est < y0s[0] - 5.0:        # not actually below the near limit
            return None
        if abs(y0_est - A) < 1e-6:
            return None
        d = B / (y0_est - A)
        if lo < d < hi:
            return float(d)
    except Exception:
        pass
    return None


def _point_b_plane_distance(model, x: float, y: float) -> Optional[float]:
    """Z distance of the 3D point where the camera ray through pixel (x, y)
    hits the B laser fan plane (None if unavailable / parallel / behind)."""
    try:
        f = next(f for f in model.fans if f.name == "B")
        cam = model.cam
        v_cam = np.array([(x - cam.cx) / cam.fx,
                          (y - cam.cy) / cam.fy, 1.0])
        v_w = cam.R @ v_cam
        denom = float(f.normal @ v_w)
        if abs(denom) < 1e-6:
            return None
        t = (f.c - float(f.normal @ cam.C)) / denom
        if t <= 0:
            return None
        P = cam.C + t * v_w
        if 0.3 < P[2] < 25.0:
            return float(P[2])
    except Exception:
        pass
    return None


def find_near_features(gray: np.ndarray, thr: float, segs: dict,
                       b_map, h_segs: Optional[list] = None,
                       min_area: int = 4, near_margin: float = 6.0,
                       model=None, laser_ds: Optional[list] = None) -> list:
    """Find bright spots / short lines inside the corridor that are NOT part
    of the measured laser lines - candidates for an obstacle that is closer
    than the main surface (e.g. the horizontal beam hitting a nearby object
    produces a dot or short horizontal line *below* the main B line).

    Region rules:
      * only inside the corridor band (between the V1 and V2 lines; central
        2/4 of the image when both are missing)
      * horizontal features in the left / right quarters are excluded
        (room edges, background)
      * vertical features in the upper / lower quarters are excluded
      * features lying on a measured laser line are excluded (they are the
        main lines / already-detected kinks themselves)
      * remaining dots / horizontal lines:
          - below the nearest measured B line (or in the bottom quarter when
            B is missing) -> 'near': closer than the main surface; distance
            estimated against the B map
          - anything else -> 'other' (reported, but not a close obstacle)

    `segs`: dict 'H','V1','V2' -> list[LineSegment] (as from measure()).
    `h_segs`: list of (ImageLine, d) of the measured B segments.
    Returns a list of dicts: kind, x, y, x0, y0, x1, y1, region, below_b,
    near, d, area.
    """
    H, W = gray.shape
    mask = (gray > thr).astype(np.uint8)
    if mask.sum() == 0:
        return []

    vlines = [s.line for s in segs.get("V1", []) + segs.get("V2", [])]

    def x_of(vl, y):
        return -(vl.b * y + vl.c) / vl.a if abs(vl.a) > 1e-9 else None

    # pixels on measured laser lines are not features
    on_line = np.zeros((H, W), bool)
    ys_g, xs_g = np.mgrid[0:H, 0:W]
    for key in ("H", "V1", "V2"):
        for s in segs.get(key, []):
            l = s.line
            on_line |= np.abs(l.a * xs_g + l.b * ys_g + l.c) <= 10.0
    mask[on_line] = 0

    # nearest measured B surface (line, d)
    h_near = None
    if h_segs:
        h_near = min(h_segs, key=lambda t: t[1])

    def y_b(x):
        if h_near is None:
            return None
        l = h_near[0]
        return -(l.a * x + l.c) / l.b if abs(l.b) > 1e-9 else None

    nlab, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, nlab):
        x0, y0, w, h, area = stats[i]
        if area < min_area:
            continue
        if w < 3 or h < 3:
            continue            # ignore sub-3x3-px artifacts (speckles)
        cx, cy = float(cents[i][0]), float(cents[i][1])
        # corridor band
        xvs = [x_of(v, cy) for v in vlines]
        xvs = [v for v in xvs if v is not None]
        if xvs:
            lo, hi = min(xvs) - 20.0, max(xvs) + 20.0
        else:
            lo, hi = 0.25 * W, 0.75 * W
        if not (lo <= cx <= hi):
            continue
        # shape classification
        if max(w, h) <= 24:
            kind = "dot"
        elif w >= 30 and h <= max(10, 0.4 * w):
            kind = "hline"
        elif h >= 30 and w <= max(10, 0.4 * h):
            kind = "vline"
        else:
            kind = "blob"
        # the B beam never reaches the top quarter; dots there are side-line
        # fragments / room features
        if kind == "dot" and cy < H / 4:
            continue
        wide = kind == "hline" or (kind == "blob" and w > h)
        tall = kind == "vline" or (kind == "blob" and h > w)
        # exclusion rules
        if wide and (cx < W / 4 or cx > 3 * W / 4):
            continue                      # horizontal in left/right quarters
        if tall and (cy < H / 4 or cy > 3 * H / 4):
            continue                      # vertical in upper/lower quarters
        yb = y_b(cx)
        below_b = (yb is not None and cy > yb + near_margin) or \
                  (yb is None and cy > 3 * H / 4)
        region = "bottom" if cy > 3 * H / 4 else ("top" if cy < H / 4 else "mid")
        near = (kind in ("dot", "hline") and below_b)
        d = None
        if near and b_map is not None:
            if kind == "dot":
                r = _point_distance_to_b(b_map, cx, cy, model=model)
                if r is not None:
                    d = r[0]
            else:
                ys_p, xs_p = np.where(lab == i)
                ln = _fit_line_to_points(xs_p.astype(float), ys_p.astype(float))
                r = b_map.fit_distance(ln)
                if r is not None and r[1] <= 10.0:
                    d = float(r[0])
            if d is None:
                # the point lies below the nearest calibrated B line
                # (closer than the map's near limit): extrapolate the
                # empirical map y0(d) = A + B/d, then the 3D ray / B-plane
                d = _extrapolate_b_distance(b_map, cx, cy)
                if d is None:
                    d = _point_b_plane_distance(model, cx, cy)
        out.append({"kind": kind, "x": cx, "y": cy, "x0": int(x0), "y0": int(y0),
                    "x1": int(x0 + w), "y1": int(y0 + h), "region": region,
                    "below_b": below_b, "near": near, "d": d,
                    "area": int(area)})
    # a chain of dots running along a side laser line is that line broken
    # into speckles, not B-beam dots -> reject the whole chain
    dots = [f for f in out if f["kind"] == "dot"]
    reject = set()
    for vl in vlines:
        chain = [f for f in dots
                 if abs(vl.a * f["x"] + vl.b * f["y"] + vl.c) <= 12.0]
        if len(chain) >= 2:
            # only an elongated chain (>= 30 px along the line) is a broken
            # laser line; a couple of isolated dots may be genuine
            tv = np.array([-vl.b, vl.a])
            t = [tv[0] * f["x"] + tv[1] * f["y"] for f in chain]
            if max(t) - min(t) >= 30.0:
                reject.update(id(f) for f in chain)
    out = [f for f in out if id(f) not in reject]
    # cross-check against the measured lasers: a genuine near obstacle also
    # kinks at least one side laser, so its distance must be supported by a
    # laser reading close to it; a lone speckle (no supporting laser) is
    # demoted to 'other', except for horizontal lines which look like real
    # sheet kinks / object edges
    if laser_ds:
        for f in out:
            if f["near"] and f["d"] is not None and f["kind"] != "hline":
                sup = any((f["d"] - 0.5) <= ld <= (f["d"] + 2.0)
                          for ld in laser_ds)
                if not sup:
                    f["near"] = False
    out.sort(key=lambda f: (f["d"] if f["d"] is not None else 99.0))
    return out


def detect_line_segments(img: np.ndarray, thr: Optional[float] = None,
                         min_h_count: int = 100, min_v_count: int = 100,
                         return_candidates: bool = False) -> dict:
    """Detect the three laser lines as lists of straight segments.

    Returns dict 'H', 'V1', 'V2' -> list[LineSegment] (possibly empty).
    With return_candidates=True also returns the list of all vertical
    candidate bands (list of (center_x, [LineSegment])) so the caller can
    reassign V1/V2 (e.g. by matching against calibration maps).
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    H, W = g.shape
    if thr is None:
        vals = np.sort(g.ravel())
        thr = max(25.0, 0.5 * (float(vals[int(0.30 * len(vals))]) + float(vals[int(0.80 * len(vals))])))
    mask = g > thr
    out = {"H": [], "V1": [], "V2": []}

    band = _h_band(mask, min_h_count)
    if band is not None:
        segs = split_segments(g, mask, band)
        keep = [s for s in segs if not s.line.is_vertical]   # horizontal-ish
        out["H"] = keep

    bands = _v_bands(mask, min_v_count)
    cands = []
    for band, cx in bands:
        segs = split_segments(g, mask, band)
        keep = [s for s in segs if s.line.is_vertical]
        if keep:
            cands.append((cx, keep))
    if len(cands) >= 1:
        out["V1"] = cands[0][1]
    if len(cands) >= 2:
        out["V2"] = cands[-1][1]
    if return_candidates:
        return out, cands
    return out


def _map_err(m: Optional["LineMap"], line: ImageLine) -> float:
    if m is None:
        return float("inf")
    r = m.fit_distance(line)
    return float("inf") if r is None else r[1]


def _fit_distance_for_fan(model: Model, fan_name: str, line_meas: ImageLine,
                          d_lo: float = 0.05, d_hi: float = 200.0) -> Optional[Tuple[float, float]]:
    """1-D fit of d for one fan given its measured image line.

    Returns (d, max_abs_residual_px) or None.
    """
    fan = next(f for f in model.fans if f.name == fan_name)

    def err(d: float) -> float:
        tr = fan.trace_on_plane(d)
        if tr is None:
            return 1e9
        pl = model.cam.image_line_from_world_line(*tr)
        if pl is None:
            return 1e9
        ap, bp, cp = pl
        am, bm, cm = line_meas.a, line_meas.b, line_meas.c
        if ap * am + bp * bm < 0:
            ap, bp, cp = -ap, -bp, -cp
        dv = np.array([bm, -am])
        c0 = np.array([-am * cm, -bm * cm])
        r = [ap * (c0[0] + s * 100 * dv[0]) + bp * (c0[1] + s * 100 * dv[1]) + cp
             for s in (-1, 1)]
        return float(np.max(np.abs(r)))

    # golden-section / parabolic search: err(d) is ~convex in d over the range
    best_d, best_e = None, 1e9
    # coarse scan
    ds = np.linspace(d_lo, d_hi, 400)
    es = np.array([err(d) for d in ds])
    i = int(np.argmin(es))
    if es[i] > 30:          # never got close: reject
        return None
    # refine: ternary search around the minimum
    a, b = ds[max(0, i - 2)], ds[min(len(ds) - 1, i + 2)]
    for _ in range(60):
        m1 = a + (b - a) / 3
        m2 = b - (b - a) / 3
        if err(m1) < err(m2):
            b = m2
        else:
            a = m1
    d = 0.5 * (a + b)
    e = err(d)
    if e > 30:
        return None
    return d, e


def measure(model: Model, maps: dict, img: np.ndarray,
            thr: Optional[float] = None,
            min_brightness: float = None,
            segs: Optional[dict] = None) -> Measurement:
    """Detect the three laser lines in `img` and return distances.

    Each laser line is split into straight segments; every segment is
    assigned the distance of the surface it reflects from (via the 1-D
    calibration maps, falling back to the 3D model).  The closest object
    distance is the minimum over all segments.  The two boundary corners
    are the nearest (V1 x B) and (V2 x B) crossing points.

    `segs` optionally supplies pre-detected segments (dict 'H','V1','V2'
    -> list[LineSegment]); by default they are detected from `img`.
    """
    if segs is None:
        segs, cands = detect_line_segments(img, thr=thr,
                                           return_candidates=True)
        # reassign V1/V2 by matching every vertical candidate group against
        # the L and R calibration maps; this rejects false bands (e.g. floor
        # traces or room edges that plain detection may pick as the rightmost
        # candidate)
        if maps and len(cands) >= 2:
            scored = []
            for cx, gl in cands:
                rep = max(gl, key=lambda s: s.n_points)
                eL = _map_err(maps.get("L"), rep.line)
                eR = _map_err(maps.get("R"), rep.line)
                scored.append((cx, gl, eL, eR))
            v1 = min(range(len(scored)), key=lambda i: scored[i][2])
            v2 = min((i for i in range(len(scored)) if i != v1),
                     key=lambda i: scored[i][3], default=None)
            if scored[v1][2] < 10:
                segs["V1"] = scored[v1][1]
            else:
                segs["V1"] = []
            if v2 is not None and scored[v2][3] < 10:
                segs["V2"] = scored[v2][1]
            else:
                segs["V2"] = []
    if min_brightness is not None:
        for k in segs:
            segs[k] = [s for s in segs[k] if s.brightness >= min_brightness]

    seg_dists = {k: [] for k in segs}      # key -> list of (LineSegment, d, err, qual)
    max_err = 15.0
    for key, fan_name in (("V1", "L"), ("V2", "R"), ("H", "B")):
        for s in segs[key]:
            res = None
            if fan_name in maps:
                r = maps[fan_name].fit_distance(s.line)
                if r is not None and r[1] <= max_err:
                    res = r
            if res is None:
                r = _fit_distance_for_fan(model, fan_name, s.line)
                if r is not None and r[1] <= max_err:
                    res = (r[0], r[1])
            if res is not None:
                d, err = float(res[0]), float(res[1])
                qual = maps[fan_name].quality(d) if fan_name in maps else 0.0
                seg_dists[key].append((s, d, err, qual))

    # beyond ~2x the calibration range the map far tails go flat and the
    # distance is no longer identifiable (grid search returns arbitrary
    # values such as 48 m or 200 m); treat those segments as missing
    if maps:
        dmax = 2.0 * max(m.range()[1] for m in maps.values())
        for k in seg_dists:
            seg_dists[k] = [t for t in seg_dists[k]
                            if 0.4 <= t[1] <= dmax]

    def closest_of(key):
        if not seg_dists[key]:
            return None
        return min(seg_dists[key], key=lambda t: t[1])

    cL, cR, cB = closest_of("V1"), closest_of("V2"), closest_of("H")

    # distance per laser = its closest segment
    distances = {}
    if cL is not None:
        distances["L"] = cL[1]
    if cR is not None:
        distances["R"] = cR[1]
    if cB is not None:
        distances["B"] = cB[1]

    # a segment is a reliable distance only if the laser still moves there
    QUAL_MIN = 4.0

    # closest object: wall-vs-object discrimination.
    #  - if the smallest reading is <= 75% of the median reading, it is a
    #    genuine local close object (the other lasers hit the background).
    #  - otherwise the scene is a flat wall and all readings should agree;
    #    the min could be a saturated-laser artifact, so take the smallest
    #    of the two best-quality readings.
    readings = []
    for fn, c in (("L", cL), ("R", cR), ("B", cB)):
        if c is not None:
            readings.append((fn, c[1], c[3]))
    if readings:
        ds_ = sorted(d for _, d, _ in readings)
        med = ds_[len(ds_) // 2]
        dmin = ds_[0]
        if len(readings) >= 2 and dmin / med <= 0.75:
            closest = dmin
        else:
            # flat scene: use the median of the readings (robust against one
            # saturated or mis-split laser)
            closest = float(np.median(ds_))
    else:
        closest = None

    # boundary corner points (image + 3D), one per side:
    #   nearest segment of the side laser (V1=left, V2=right) crossed with
    #   the nearest H (bottom/floor) segment containing the intersection.
    #   3D depth uses the bottom laser's distance (it stays informative
    #   longest); the 3D point lies on fan_k * fan_B * {z = depth}.
    corners = None
    corners_3d = None
    pairs = []
    for vk, dk in (("V1", "L"), ("V2", "R")):
        if not seg_dists[vk] or not seg_dists["H"]:
            continue
        sv, dv, ev, qv = min(seg_dists[vk], key=lambda t: t[1])
        cand = None
        for sh, dh, eh, qh in seg_dists["H"]:
            p2 = line_intersection(sv.line, sh.line)
            if p2 is None:
                continue
            if not (sv.contains_point(p2) and sh.contains_point(p2)):
                continue
            if cand is None or dh < cand[2]:
                cand = (p2, sh, dh, qh)
        if cand is None:
            continue
        p2, sh, dh, qh = cand
        d3 = dh if qh >= QUAL_MIN else (dv if qv >= QUAL_MIN else None)
        if d3 is None:
            continue
        fan_v = next(f for f in model.fans if f.name == dk)
        fan_h = next(f for f in model.fans if f.name == "B")
        p3 = _fan_intersection_3d(fan_v, fan_h, d3)
        pairs.append((p2, p3))
    if pairs:
        corners = tuple(p[0] for p in pairs)
        corners_3d = tuple(p[1] for p in pairs)

    # representative single ImageLine per laser for reporting
    lines = {}
    for key in ("H", "V1", "V2"):
        if segs[key]:
            # brightest / longest segment as representative
            rep = max(segs[key], key=lambda s: s.n_points)
            lines[key] = rep.line
        else:
            lines[key] = ImageLine(ok=False)

    details = {}
    for key, fan_name in (("V1", "L"), ("V2", "R"), ("H", "B")):
        details[fan_name] = [
            {"d": d, "err_px": e, "quality": q, "n": s.n_points,
             "brightness": s.brightness, "midpoint": s.midpoint()}
            for (s, d, e, q) in seg_dists[key]
        ]

    res = Measurement(distances=distances, closest=closest,
                      corners=corners, corners_3d=corners_3d,
                      lines=lines, details=details)
    res._segs = segs
    return res


def _fan_intersection_3d(fan1: Fan, fan2: Fan, d: float):
    """3D point = fan1 ∩ fan2 ∩ plane Z=d."""
    A = np.array([fan1.normal, fan2.normal, [0.0, 0.0, 1.0]])
    b = np.array([fan1.c, fan2.c, d])
    try:
        return tuple(np.linalg.solve(A, b))
    except np.linalg.LinAlgError:
        return (np.nan, np.nan, np.nan)


# ----------------------------------------------------------------------------
# calibration file I/O
# ----------------------------------------------------------------------------

def save_calibration(model: Model, path: str, maps: Optional[dict] = None,
                     report: Optional[dict] = None, meta: Optional[dict] = None) -> None:
    doc = {"version": 1, "model": model.as_dict()}
    if maps is not None:
        doc["maps"] = {k: m.as_dict() for k, m in maps.items()}
    if report is not None:
        doc["report"] = report
    if meta is not None:
        doc["meta"] = meta
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def load_calibration(path: str) -> Tuple[Model, dict, dict]:
    """Returns (model, doc, maps)."""
    with open(path) as f:
        doc = json.load(f)
    model = Model.from_dict(doc["model"])
    maps = {k: LineMap.from_dict(v) for k, v in doc.get("maps", {}).items()}
    return model, doc, maps
