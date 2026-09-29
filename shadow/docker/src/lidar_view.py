#!/usr/bin/env python3
"""
lidar_view.py — visualise the hackathon LiDAR bags
==================================================

Data facts (reverse-engineered from the bags):
  * each frame is a point cloud in the VEHICLE frame:
        x  = longitudinal,  metres;  FORWARD = -x
           (beams start at x ~ -1.15 m, a 1.15 m dead zone)
        y  = lateral,  metres  (+y assumed left)
  * the only per-point extra channel is a 0..255 value in the
    "z" slot  (an intensity-like quantity);  the declared
    ring/timestamp slots contain driver junk,  so NO height is
    available in this dataset.  The clouds are therefore 2.5D:
    (forward distance,  lateral) + intensity.

Rendering per frame  (one 1600x720 frame -> one video per bag):
  * LEFT   : TOP view — bird's-eye x/y,  forward (-x) up,
             points coloured by intensity
  * RIGHT  : FORWARD view — looking along -x:
             horizontal = lateral y,  vertical = forward distance
             with perspective compression (near = bottom,
             far -> horizon),  coloured by intensity

Usage
-----
    python3 lidar_view.py                 # all bags
    python3 lidar_view.py doubleT_obstacle
    python3 lidar_view.py --probe         # dump frame stats only
"""

import glob
import os
import sqlite3
import struct
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BAG_DIR = os.path.join(HERE, "датасет (1)", "archive", "for_hackathon")

# --------------------------------------------------------------------
# PointCloud2 parsing  (non-standard CDR:  field names include the
# NUL in their length,  datatype stored as a 4-byte int,  the data
# array starts 200 bytes from the end of the blob,  pstep = 26)
# --------------------------------------------------------------------

def _align(i, base=4):
    return i + ((-((i - base) % 4)) % 4)


def parse_pc2(blob: np.ndarray) -> dict:
    U32 = lambda o: struct.unpack_from("<I", blob, o)[0]
    i = 4                                   # skip CDR header
    i += 8                                  # stamp
    ln = U32(i)
    i += 4
    fid = blob[i:i + ln].tobytes().rstrip(b"\x00").decode("latin1")
    i += ln
    i = _align(i)
    h = U32(i)
    w = U32(i + 4)
    nf = U32(i + 8)
    i += 12
    DT_CODE = {1: "u1", 2: "u2", 3: "i2", 4: "u4",
               5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    DT_SIZE = {1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 4, 8: 8}
    fields = []
    for _ in range(nf):
        flen = U32(i)
        i += 4
        name = blob[i:i + flen].tobytes().rstrip(b"\x00").decode("latin1")
        i += flen
        i = _align(i)
        off = U32(i)
        dt = U32(i + 4)
        i += 12
        fields.append((name, off, DT_SIZE[dt], DT_CODE[dt]))
    pstep = max(off + size for _, off, size, _ in fields)
    data = blob[-(len(blob) - 200):]
    n = len(data) // pstep
    arr = np.frombuffer(data, dtype=np.uint8, count=n * pstep)
    arr = arr.reshape(n, pstep)
    out = {"fid": fid, "n": n, "fields": {}}
    for name, off, size, code in fields:
        out["fields"][name] = arr[:, off:off + size].astype(
            np.uint8).view(code)
    return out


def load_bag(path: str):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    for ts, blob in con.execute(
            "SELECT timestamp, data FROM messages ORDER BY timestamp"):
        yield ts, parse_pc2(np.frombuffer(blob, dtype=np.uint8))
    con.close()


def pc_data(pc: dict):
    """(d_fwd, y, intensity) with invalid/zero points removed.

    d_fwd = -x  :  forward distance in front of the vehicle [m]
    """
    f = pc["fields"]
    x = np.ascontiguousarray(f["x"].reshape(-1, 1))
    y = np.ascontiguousarray(f["y"].reshape(-1, 1))
    s = np.ascontiguousarray(f["z"].reshape(-1, 1))   # intensity slot
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    # zero-padded empty beams
    keep &= ~((x == 0) & (y == 0) & (s == 0))
    d = -x[keep]
    return d, y[keep], s[keep].astype(np.float32)


# --------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------

BEV_HALF = 22.0        # metres shown each side  (forward/back, left/right)
BEV_PX = 720           # panel size
FWD_W = 880
FWD_H = 720
FWD_DMAX = 40.0        # metres shown in the forward view


def _bw():
    """plain white point  (black-and-white mode:  geometry first)"""
    return (255, 255, 255)


def bev_frame(d, y, s):
    """Top view.  d (forward) up,  y left.  d_fwd = -x."""
    ppm = BEV_PX / (2 * BEV_HALF)           # px per metre
    img = np.zeros((BEV_PX, BEV_PX, 3), np.uint8)
    m = (d >= -BEV_HALF) & (d <= BEV_HALF) & (np.abs(y) <= BEV_HALF)
    d_, y_, s_ = d[m], y[m], s[m]
    u = np.clip((BEV_PX / 2 + y_ * ppm).astype(np.int32), 0, BEV_PX - 1)
    v = np.clip((BEV_PX / 2 - d_ * ppm).astype(np.int32), 0, BEV_PX - 1)
    # nearest-wins per pixel  (densest part of the cluster)
    key = v * BEV_PX + u
    best = np.full(BEV_PX * BEV_PX, 1e9, np.float32)
    np.minimum.at(best, key, np.abs(d_))
    order = np.argsort(key)
    dd = d_[order]
    bk = best[key[order]]
    sel = order[dd <= bk]
    img[v[order][sel], u[order][sel]] = _bw()
    # grid every 5 m  (dark grey)
    c = BEV_PX // 2
    step = int(5 * ppm)
    for k in range(-int(BEV_HALF) // 5, int(BEV_HALF) // 5 + 1):
        p = c + k * step
        if 0 <= p < BEV_PX:
            cv2.line(img, (p, 0), (p, BEV_PX), (35, 35, 35), 1)
            cv2.line(img, (0, p), (BEV_PX, p), (35, 35, 35), 1)
    for r in (5, 10, 15, 20):
        cv2.circle(img, (c, c), int(r * ppm), (50, 50, 50), 1)
    # ego marker,  forward up
    cv2.arrowedLine(img, (c, c), (c, c - 40), (255, 255, 255), 1,
                    tipLength=0.3)
    cv2.circle(img, (c, c), 3, (255, 255, 255), -1)
    cv2.putText(img, "TOP  (forward up)", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "5m", (c + 3, c - step - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    return img


def fwd_frame(d, y, s):
    """Forward view with perspective compression.

    vertical:  d=0 at the bottom,  d->inf at the horizon line
               (45% down from the top);
    horizontal:  lateral y  (left = +y).
    """
    img = np.zeros((FWD_H, FWD_W, 3), np.uint8)
    D0 = 6.0
    m = (d > 0.3) & (d < FWD_DMAX) & (np.abs(y) < 12.0)
    d_, y_, s_ = d[m], y[m], s[m]
    # perspective-ish row mapping
    t = d_ / (d_ + D0)                       # 0..1,  compressed at far
    horizon = int(0.45 * FWD_H)
    v = np.clip((FWD_H - 1 - t * (FWD_H - 1 - horizon)).astype(np.int32),
                0, FWD_H - 1)
    u = np.clip((FWD_W / 2 - y_ * (FWD_W / 2) / 12.0).astype(np.int32),
                0, FWD_W - 1)
    key = v * FWD_W + u
    best = np.full(FWD_W * FWD_H, 1e9, np.float32)
    np.minimum.at(best, key, d_)
    order = np.argsort(key)
    dd = d_[order]
    bk = best[key[order]]
    sel = order[dd <= bk]
    img[v[order][sel], u[order][sel]] = _bw()
    # horizon
    cv2.line(img, (0, horizon), (FWD_W, horizon), (60, 60, 60), 1)
    # distance rows:  2, 5, 10, 20, 40 m
    for ddv in (2, 5, 10, 20, 40):
        tt = ddv / (ddv + D0)
        vv = int(FWD_H - 1 - tt * (FWD_H - 1 - horizon))
        cv2.line(img, (0, vv), (FWD_W, vv), (50, 50, 50), 1)
        cv2.putText(img, f"{ddv}m", (4, vv - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    # lateral centre line
    cv2.line(img, (FWD_W // 2, horizon), (FWD_W // 2, FWD_H - 1),
             (40, 40, 40), 1)
    cv2.putText(img, "FORWARD  (looking -x)", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def probe(bag_path):
    n = 0
    ts0 = None
    for ts, pc in load_bag(bag_path):
        d, y, s = pc_data(pc)
        if n == 0:
            print(f"{os.path.basename(os.path.dirname(bag_path))}: "
                  f"fid={pc['fid']!r}  raw={pc['n']}  valid={len(d)}  "
                  f"fid2={pc['fields'].keys()}")
            print(f"  fwd d:[{d.min():.2f},{d.max():.2f}] med={np.median(d):.2f}"
                  f"   y:[{y.min():.2f},{y.max():.2f}]"
                  f"   s med={np.median(s):.0f}")
        n += 1
        ts0 = ts if n == 1 else ts0
    print(f"  frames={n}  duration={(ts - ts0)/1e9:.1f}s")


def render_bag(bag_path, out_path):
    msgs = list(load_bag(bag_path))
    if not msgs:
        print("  (no messages)")
        return
    fps = (len(msgs) - 1) / max(1e-9, (msgs[-1][0] - msgs[0][0]) / 1e9)
    fps = min(max(fps, 1.0), 30.0)
    W, H = 1600, 720
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    print(f"  {len(msgs)} frames  @ {fps:.1f} fps  ->  {out_path}")
    bagname = os.path.basename(os.path.dirname(bag_path))
    for i, (ts, pc) in enumerate(msgs):
        d, y, s = pc_data(pc)
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:, :BEV_PX] = bev_frame(d, y, s)
        frame[:, BEV_PX:BEV_PX + FWD_W] = fwd_frame(d, y, s)
        t_rel = (ts - msgs[0][0]) / 1e9
        cv2.putText(frame, bagname, (BEV_PX + 10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"frame {i}/{len(msgs) - 1}   "
                           f"t={t_rel:6.1f}s   pts={len(d)}",
                    (BEV_PX + 10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        writer.write(frame)
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(msgs)}")
    writer.release()


def main():
    args = [a for a in sys.argv[1:]]
    if "--probe" in args:
        args.remove("--probe")
        bags = args or sorted(glob.glob(os.path.join(BAG_DIR, "*/")))
        for b in bags:
            for db in glob.glob(os.path.join(b, "*.db3")):
                probe(db)
        return
    names = args
    if not names:
        names = sorted(
            os.path.basename(b.rstrip("/"))
            for b in glob.glob(os.path.join(BAG_DIR, "*/")))
    for name in names:
        db = glob.glob(os.path.join(BAG_DIR, name, "*.db3"))
        if not db:
            print(f"!! no db3 for {name}")
            continue
        out = os.path.join(HERE, f"lidar_view_{name}.mp4")
        print(f"== {name}")
        render_bag(db[0], out)


if __name__ == "__main__":
    main()
