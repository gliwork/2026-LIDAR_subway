#!/usr/bin/env python3
"""
summarise.py - one overview table across all scenario runs.

Reads out/<scenario>/objects.csv for every scenario and writes
out/ALL_objects.csv: every consolidated object (all scenarios),
sorted by scenario then persistence.  Also prints a compact per-
scenario digest to stdout.

    /usr/bin/python3.10 summarise.py
"""
import csv
import glob
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    rows = []
    digest = {}
    for path in sorted(glob.glob(os.path.join(OUT, "*", "objects.csv"))):
        scen = os.path.basename(os.path.dirname(path))
        with open(path, newline="") as f:
            rs = list(csv.DictReader(f))
        digest[scen] = (len(rs),
                        sum(1 for r in rs if r.get("kind") == "STABLE"))
        for r in rs:
            r2 = dict(r)
            r2["scenario"] = scen
            rows.append(r2)
    rows.sort(key=lambda r: (r["scenario"],
                             -int(r.get("seen_in_frames", 0)),
                             int(r.get("chunk", 0))))
    out_path = os.path.join(OUT, "ALL_objects.csv")
    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} objects)\n")
    print(f"{'scenario':<40} {'objects':>8} {'stable':>8}")
    for scen, (n, ns) in digest.items():
        print(f"{scen:<40} {n:>8} {ns:>8}")
    print()
    # top stable objects per scenario
    for scen in digest:
        sc = [r for r in rows if r["scenario"] == scen
              and r.get("kind") == "STABLE"]
        if not sc:
            continue
        sc.sort(key=lambda r: -float(r["max_size"]))
        print(f"== {scen}: {len(sc)} stable, biggest 5 by size:")
        for r in sc[:5]:
            print(f"   chunk {int(r['chunk']):2d} "
                  f"({float(r['d_fwd']):6.1f} m ahead)  x={r['cx']:>6} "
                  f"z={r['cz']:>6}  size={r['max_size']:>5} m  "
                  f"pts={r['npts']:>6}  seen {r['seen_in_frames']} frames")
        print()


if __name__ == "__main__":
    main()
