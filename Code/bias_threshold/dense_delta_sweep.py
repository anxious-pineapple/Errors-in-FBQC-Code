"""Threshold vs slot bias δ at twice the δ resolution of the notebooks.

Fills in the δ values between the notebooks' points (p_meas: steps of 1/16
instead of 1/8; p_erase: steps of 1/8 instead of 1/4) for the two regular-fusion
circuits only -- the unmodified hexagon code and HexagonMOD2's §4 plain circuit.
No H-fusion circuit is run.

The sweeps are each notebook's own code: the definition cells are executed from
the .ipynb (long-running loops and plots are skipped), and the notebook's own
`bias_meas_p_sweep` / `bias_erase(sure)_p_sweep`, grids, seeds and crossing
function are called at the new δ only. The notebooks' existing crossings are
copied in from bias_threshold/*_threshold_vs_delta.csv, so the output CSVs hold
the full δ grid.

One addition to the notebooks' two-pass method: if the fine grid (centred on the
12 x 18 crossing) contains no crossing for a pair, that pair gets its own coarse
pass at its own two sizes and a fine grid around that. Such crossings are
flagged `fallback` in the CSV. This is also run on the hexagon notebook's δ = 1
p_meas point, whose 18 x 24 crossing was never found.

Usage (from Code/):  python bias_threshold/dense_delta_sweep.py {hexagon|hexagonmod2}
"""
import json
import os
import sys
import time
import types

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)

# (cell index, cut marker): exec the cell's source up to the first line starting
# with the marker, or the whole cell if None.
CELLS = {
    "hexagonmod2": ("HexagonMOD2_fbqc_walkthrough.ipynb", [
        (1, None), (3, None), (10, None), (12, None), (14, None), (18, None), (20, None),
        (22, None), (24, None), (26, None), (28, None), (37, "CROSSINGS = {}"), (41, None),
        (43, None), (49, None), (50, None), (56, None),
        (65, "th_vs_delta = {}"), (70, "th_vs_delta_e = {}"),
    ]),
    "hexagon": ("Hexagon_fbqc_walkthrough.ipynb", [
        (1, None), (3, None), (14, None), (18, None), (19, None), (21, None), (29, None),
        (31, None), (33, None), (35, None), (37, None), (41, None), (44, None),
        (47, "print(\"Threshold estimate"), (57, None), (64, "# Regression check"),
        (76, None), (82, None), (86, "t_h0 = time.time()"), (98, None), (108, None),
        (117, "th_vs_delta = {\"meas\""), (121, "th_vs_delta[\"erasure\"]"),
    ]),
}

DENSE = {"meas": np.round(np.linspace(0, 1, 17), 4), "erasure": np.round(np.linspace(0, 1, 9), 4)}
PAIRS = ((4, 6), (6, 8))          # x_max pairs: Lx 12 x 18, 18 x 24


def load_notebook(name):
    import nbformat
    path, cells = CELLS[name]
    nb = nbformat.read(os.path.join(CODE_DIR, path), as_version=4)
    # a registered module, so the notebooks' forked pool workers can be pickled by name
    mod = types.ModuleType("__nb__")
    sys.modules["__nb__"] = mod
    ns = mod.__dict__
    for idx, cut in cells:
        src = nb.cells[idx].source
        if cut is not None:
            lines = src.split("\n")
            stop = next(i for i, l in enumerate(lines) if l.startswith(cut))
            src = "\n".join(lines[:stop])
        t0 = time.time()
        exec(compile(src, f"<{name} cell {idx}>", "exec"), ns)
        print(f"[{name}] cell {idx} ok ({time.time() - t0:.0f}s)", flush=True)
    return ns


def setup(name, ns):
    """(per-noise dict of sweep, coarse grid, fine span), crossing function, circuit label."""
    if name == "hexagonmod2":
        circ = "§4 plain"
        cfg = {"meas": (lambda x, d, ps: ns["bias_meas_p_sweep"](x, circ, d, ps),
                        ns["TH_COARSE_PS"], ns["TH_FINE_SPAN"]),
               "erasure": (lambda x, d, ps: ns["bias_erase_p_sweep"](x, circ, d, ps),
                           ns["TH_COARSE_PS_E"], ns["TH_FINE_SPAN_E"])}
        return cfg, ns["crossing"], circ
    circ = "regular fusion"
    cfg = {"meas": (lambda x, d, ps: ns["bias_meas_p_sweep"](x, circ, d, ps),
                    ns["TH_COARSE_PS"]["meas"], ns["TH_FINE_SPAN"]["meas"]),
           "erasure": (lambda x, d, ps: ns["bias_erasure_p_sweep"](x, circ, d, ps),
                       ns["TH_COARSE_PS"]["erasure"], ns["TH_FINE_SPAN"]["erasure"])}
    return cfg, ns["find_crossing"], circ


def two_pass(sweep, coarse_ps, span, cross, delta, pairs=(0, 1), log=None):
    """The notebooks' two-pass threshold, plus a per-pair fallback (see module doc).
    Returns {pair index: (crossing or None, source)}."""
    out = {}
    coarse = {x: sweep(x, delta, coarse_ps) for x in (4, 6)}
    centre = cross(coarse_ps, coarse[4], coarse[6])
    if log is not None:
        log.append(dict(delta=float(delta), stage="coarse", sizes=[4, 6], ps=list(map(float, coarse_ps)),
                        curves=[list(map(float, coarse[x])) for x in (4, 6)]))
    if centre is not None:
        ps = np.round(centre * np.linspace(*span), 6)
        fine = {x: sweep(x, delta, ps) for x in (4, 6, 8)}
        if log is not None:
            log.append(dict(delta=float(delta), stage="fine", sizes=[4, 6, 8], ps=list(map(float, ps)),
                            curves=[list(map(float, fine[x])) for x in (4, 6, 8)]))
        for k in (0, 1):     # both, even if only one is wanted: the other checks reproduction
            a, b = PAIRS[k]
            out[k] = (cross(ps, fine[a], fine[b]), "two-pass")
    for k in pairs:
        if out.get(k, (None,))[0] is not None:
            continue
        a, b = PAIRS[k]
        cc = {x: (coarse[x] if x in coarse else sweep(x, delta, coarse_ps)) for x in (a, b)}
        c_k = cross(coarse_ps, cc[a], cc[b])
        if log is not None:
            log.append(dict(delta=float(delta), stage=f"fallback-coarse", sizes=[a, b],
                            ps=list(map(float, coarse_ps)), curves=[list(map(float, cc[x])) for x in (a, b)]))
        if c_k is None:
            out[k] = (None, "fallback: no coarse crossing")
            continue
        ps = np.round(c_k * np.linspace(*span), 6)
        ff = {x: sweep(x, delta, ps) for x in (a, b)}
        if log is not None:
            log.append(dict(delta=float(delta), stage="fallback-fine", sizes=[a, b], ps=list(map(float, ps)),
                            curves=[list(map(float, ff[x])) for x in (a, b)]))
        out[k] = (cross(ps, ff[a], ff[b]), "fallback")
    return out


def read_csv(path, circuit):
    """{delta: row} for one circuit (a file may hold several at the same δ)."""
    rows = {}
    with open(path) as f:
        header = next(f).rstrip("\n").split(",")
        for line in f:
            vals = dict(zip(header, line.rstrip("\n").split(",")))
            if vals["circuit"] == circuit:
                rows[float(vals["delta"])] = vals
    return rows


def write_csv(path, rows):
    cols = ["circuit", "delta", "cross_12x18", "cross_18x24", "source_12x18", "source_18x24"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for d in sorted(rows):
            f.write(",".join(str(rows[d].get(c, "")) for c in cols) + "\n")


def main(name):
    os.chdir(CODE_DIR)
    ns = load_notebook(name)
    cfg, cross, circ = setup(name, ns)
    for noise in ("meas", "erasure"):
        sweep, coarse_ps, span = cfg[noise]
        src = os.path.join(HERE, f"{name}_{noise}_threshold_vs_delta.csv")
        dst = os.path.join(HERE, f"{name}_{noise}_threshold_vs_delta_dense.csv")
        logp = os.path.join(HERE, f"{name}_{noise}_threshold_vs_delta_dense_curves.jsonl")
        old = read_csv(src, circ)
        rows = read_csv(dst, circ) if os.path.exists(dst) else {}
        for d, r in old.items():
            if d not in rows:
                rows[d] = dict(circuit=circ, delta=d, cross_12x18=r["cross_12x18"], cross_18x24=r["cross_18x24"],
                               source_12x18="notebook" if r["cross_12x18"] else "",
                               source_18x24="notebook" if r["cross_18x24"] else "")
        write_csv(dst, rows)

        todo = [(float(d), (0, 1)) for d in DENSE[noise] if float(d) not in rows]
        # notebook points with a missing crossing: retry just that pair with the fallback
        todo += [(d, tuple(k for k, c in enumerate(("cross_12x18", "cross_18x24")) if not rows[d][c]))
                 for d in sorted(rows) if not (rows[d]["cross_12x18"] and rows[d]["cross_18x24"])
                 and rows[d].get("source_12x18") == "notebook"]
        for d, pairs in todo:
            t0, log = time.time(), []
            res = two_pass(sweep, coarse_ps, span, cross, d, pairs=pairs, log=log)
            r = rows.setdefault(d, dict(circuit=circ, delta=d))
            for k, col in enumerate(("12x18", "18x24")):
                if k not in pairs and res.get(k, (None,))[0] is not None:
                    print(f"[{name}] {noise} δ={d:.4f}: recomputed {col} {res[k][0] * 100:.3f}% "
                          f"vs notebook {float(r[f'cross_{col}']) * 100:.3f}%  (reproduction check)", flush=True)
                if k in pairs:
                    x, how = res[k]
                    r[f"cross_{col}"] = "" if x is None else f"{x:.6g}"
                    r[f"source_{col}"] = how
            write_csv(dst, rows)
            with open(logp, "a") as f:
                for entry in log:
                    f.write(json.dumps(entry) + "\n")
            fmt = lambda c: f"{float(r[c]) * 100:.3f}%" if r.get(c) else "--"
            print(f"[{name}] {noise} δ={d:.4f}: 12x18 {fmt('cross_12x18')} ({r.get('source_12x18')})  "
                  f"18x24 {fmt('cross_18x24')} ({r.get('source_18x24')})  ({time.time() - t0:.0f}s)", flush=True)
    print(f"[{name}] done", flush=True)


if __name__ == "__main__":
    if "--load-only" in sys.argv:
        os.chdir(CODE_DIR)
        setup(sys.argv[1], load_notebook(sys.argv[1]))
    else:
        main(sys.argv[1])
