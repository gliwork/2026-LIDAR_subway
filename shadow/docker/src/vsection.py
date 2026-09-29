#!/usr/bin/env python3
"""
vsection  —  "vertical"  cross  sections  of  the  corridor  using  the
0-255  point  attribute  (  the  "s"  slot  of  the  26-byte  record  )  as
the  ROW  axis  :

        image  size  =  [ 256  rows  (  s  )  ]  x  [  N_Y  columns  (  y  )  ]

For  every  D  in  2..180 m  the  slab  f  =  -x  in  [D  -  0.05,  D]  is
binned  into  ( s ,  y  )  :  row  =  s  (  0  at  the  bottom,  255  at  the
top  ),  column  =  lateral  y  (  -yspan  ..  +yspan  ).  B/W  :  white
points  on  black,  log  scale.

WHAT  THE  s  CHANNEL  IS  (  forensics  results  ,  see  chat  )  :

  *  values  0..255,  strongly  concentrated  at  2..13  (  floor  cluster
    ~  4..6  ),  rare  tail  up  to  255  (  0.1-1%  of  points  )  ;
  *  it  is  NOT  a  linear  height  in  metres  :  a  3-D  ray
    collinearity  test  (  z  =  c*s  +  d  )  finds  no  (c,  d)  that
    makes  same-  (  theta,  s  )  point  cells  collinear  with  the
    sensor  origin  —  any  c  only  increases  the  within-cell  angular
    spread  (  monotonically  )  ;
  *  it  is  NOT  a  beam  pitch  index  either  (  z  =  r2*tan(a*s  +  b)  :
    the  minimum  sits  at  a  =  0  ,  i.e.  no  angle  structure  )  ;
  *  s  does  correlate  weakly  with  "  how  low  /  how  central  "  :
    floor  points  cluster  at  low  s  ,  walls  spread  over  0..20  ,
    ceiling-like  points  are  the  rare  s  >  60  tail  .

So  the  row  axis  is  an  uncalibrated  "  verticalness  index  ".  If  the
data  producer  documents  the  byte  (  e.g.  1  unit  =  1  cm  height  ,
255  =  clipped  ceiling  )  the  same  images  immediately  become  metric
cross  sections  —  only  the  row  label  would  change.

Outputs  per  bag  (  in  tunnel_section/  )  :
    vsec_<bag>_D<nn>.png        one  [256,  N_Y]  image  per  D
    vsheet_<bag>_<k>.png       sheets  of  24  such  images  (  4  x  6  )

Usage:
    python3  vsection.py  [  bag  name  filter  ...  ]
                          --nstack  3  --step  10  --slice  0.05
                          --yspan  4  --sheets  4
"""
import sys, os, sqlite3
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
BAGS = os.path.join(HERE, "датасет (1)", "archive", "for_hackathon")
OUT = os.path.join(HERE, "tunnel_section")

D_MIN, D_MAX, D_STEP = 2.0, 180.0, 2.0
SLICE_M = 0.05
Y_SPAN_DEFAULT = 4.0    #  +-  m
Y_BIN = 0.1             #  m  /  column
S_ROWS = 256

CHARS = " .:-=+*#%@"

def pc_data(blob):
    p = np.frombuffer(blob, np.uint8)[200:].reshape(-1, 26)
    x = p[:, 0:4].view(np.float32).reshape(-1)
    y = p[:, 4:8].view(np.float32).reshape(-1)
    s = p[:, 8:12].view(np.float32).reshape(-1)
    k = (np.hypot(x, y) > 0.3) & np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    return x[k], y[k], s[k]

def build_vsection(msgs, nstack, step, D_list, slice_m, yspan, n_y):
    """(  len(D_list),  256,  n_y  )  int  counts  :  slab  on  f  ."""
    grids = {d: np.zeros((S_ROWS, n_y), np.int32) for d in D_list}
    used = 0
    for j in range(0, len(msgs), step):
        if used >= nstack:
            break
        x, y, s = pc_data(msgs[j][0])
        used += 1
        f = -x
        for d in D_list:
            m = (f >= d - slice_m) & (f <= d) & (np.abs(y) < yspan)
            if not m.any():
                continue
            si = np.clip(np.round(s[m]), 0, 255).astype(np.int32)
            bi = ((y[m] + yspan) / Y_BIN + 1e-6).astype(np.int32)
            ok = (bi >= 0) & (bi < n_y)
            idx = si[ok] * n_y + bi[ok]
            grids[d] += np.bincount(idx, minlength=S_ROWS * n_y).reshape(S_ROWS, n_y)
    return grids

def render(grid, D, save_path, yspan, n_y, scale=4):
    """[256,  n_y]  log-scale  B/W  image,  s  =  0  at  the  BOTTOM  ."""
    g = np.log1p(grid.astype(np.float32))
    mx = g.max()
    img = np.zeros((S_ROWS, n_y), np.uint8)
    if mx > 0:
        img = (255 * g / mx).astype(np.uint8)
    img = img[::-1]                       #  s  =  255  at  the  top  of  the  image
    img = cv2.resize(img, (n_y * scale, S_ROWS * scale), interpolation=cv2.INTER_NEAREST)
    #  axis  ticks
    cv2.putText(img, f"D  =  {D:3.0f}  m   (  rows  =  s  0..255,  bottom  =  0  )",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1, cv2.LINE_AA)
    cv2.putText(img, f"y:  -{yspan:.0f}  ..  +{yspan:.0f}  m", (8, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
    for sv in (0, 50, 100, 150, 200, 255):
        ypx = int((S_ROWS - 1 - sv) * scale) + 4
        cv2.line(img, (0, ypx), (14, ypx), 200, 1)
        cv2.putText(img, str(sv), (18, ypx + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 200, 1, cv2.LINE_AA)
    cv2.imwrite(save_path, img)

def render_sheet(grids, D_list, bag, sheet_no, per_sheet, save_path,
                 yspan, n_y, cscale=1):
    """4  x  6  grid  of  [256,  n_y]  cells  (  scaled  down  )  ."""
    cell_w = int(n_y * cscale)
    cell_h = int(S_ROWS * cscale)
    cols, rows = 4, 6
    margin = 26
    W = cols * (cell_w + margin) + margin
    H = rows * (cell_h + margin) + margin
    sheet = np.zeros((H, W), np.uint8)
    for k in range(per_sheet):
        gi = sheet_no * per_sheet + k
        if gi >= len(D_list):
            break
        d = D_list[gi]
        g = np.log1p(grids[d].astype(np.float32))
        mx = g.max()
        cell = np.zeros((S_ROWS, n_y), np.uint8)
        if mx > 0:
            cell = (255 * g / mx).astype(np.uint8)
        cell = cell[::-1]
        if cscale < 1:
            cell = cv2.resize(cell, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        r, c = divmod(k, cols)
        x0 = margin + c * (cell_w + margin)
        y0 = margin + r * (cell_h + margin)
        sheet[y0:y0 + cell_h, x0:x0 + cell_w] = cell
        cv2.putText(sheet, f"D  =  {d:3.0f}  m", (x0, y0 + cell_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
    cv2.putText(sheet, f"{bag}   —   vertical  cross  sections,  rows  =  s  (  0..255  ),  "
                       f"columns  =  y  (  m  )", (margin, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)
    cv2.imwrite(save_path, sheet)

def process_bag(path, nstack, step, slice_m, yspan, sheets):
    bag = os.path.basename(os.path.dirname(path))
    n_y = int(2 * yspan / Y_BIN) + 1
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    msgs = con.execute("SELECT data FROM messages ORDER BY timestamp").fetchall()
    con.close()
    D_list = [d for d in np.arange(D_MIN, D_MAX + 0.01, D_STEP)]
    print(f"--  {bag}:  {len(msgs)}  msgs,  {len(D_list)}  D  values  ...", flush=True)
    grids = build_vsection(msgs, nstack, step, D_list, slice_m, yspan, n_y)

    #  per-D  images
    for d in D_list:
        render(grids[d], d, os.path.join(OUT, f"vsec_{bag}_D{int(d):03d}.png"),
               yspan, n_y)
    #  sheets
    per = 24
    for sh in range(int(np.ceil(len(D_list) / per))):
        if sheets and sh >= sheets:
            break
        render_sheet(grids, D_list, bag, sh, per,
                     os.path.join(OUT, f"vsheet_{bag}_{sh+1}.png"), yspan, n_y,
                     cscale=0.75)

    #  a  few  facts  about  s  for  the  log  (  over  all  D  slabs  )
    tot = int(sum(int(g.sum()) for g in grids.values()))
    lo = int(sum(int(g[:14].sum()) for g in grids.values()))
    hi = int(sum(int(g[60:].sum()) for g in grids.values()))
    print(f"   s  distribution  (  slabs  )  :  s<14  =  {lo / max(tot, 1) * 100:.0f}%,  "
          f"s>60  =  {hi / max(tot, 1) * 100:.1f}%  of  all  points", flush=True)
    print(f"==  {bag}:  {len(D_list)}  vsec  images  +  sheets  ->  {OUT}/", flush=True)

def main():
    a = sys.argv[1:]
    kw = {}
    bags = []
    i = 0
    while i < len(a):
        if a[i] in ("--nstack", "--step", "--slice", "--yspan", "--sheets"):
            kw[a[i][2:]] = float(a[i + 1]) if a[i] != "--sheets" else int(a[i + 1])
            i += 2
        else:
            bags.append(a[i])
            i += 1
    nstack = int(kw.get("nstack", 3))
    step = int(kw.get("step", 10))
    slice_m = float(kw.get("slice", SLICE_M))
    yspan = float(kw.get("yspan", Y_SPAN_DEFAULT))
    os.makedirs(OUT, exist_ok=True)

    if not bags:
        bags = sorted(os.listdir(BAGS))
    else:
        allb = sorted(os.listdir(BAGS))
        bags = [b for b in allb if any(f in b for f in bags)]

    for b in bags:
        p = os.path.join(BAGS, b)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith(".db3"):
                process_bag(os.path.join(p, f), int(nstack), step, slice_m, yspan,
                            kw.get("sheets", 0))

if __name__ == "__main__":
    main()
