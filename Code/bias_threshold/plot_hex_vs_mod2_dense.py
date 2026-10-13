"""Threshold vs slot bias δ: unmodified hexagon code vs HexagonMOD2 (§4 plain),
no H-fusion. Reads the *_dense.csv files written by dense_delta_sweep.py.

Each point is the mean of the 12 x 18 and 18 x 24 crossings and the bar spans the
two, as in HexagonMOD2_fbqc_walkthrough.ipynb §14. A crossing found by the
fallback pass (the 18 x 24 crossing lay outside the fine grid centred on the
12 x 18 one) is drawn with an open marker.

Usage (from Code/):  python bias_threshold/plot_hex_vs_mod2_dense.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, file prefix, colour, line style, marker) -- §14's colours for the two codes
SERIES = [("hexagon, regular fusion", "hexagon", "#1baf7a", "-", "o"),
          ("HexagonMOD2, §4 plain", "hexagonmod2", "#2a78d6", "--", "s")]
PANELS = [("meas", "measurement error", "$p_\\mathrm{meas}$"),
          ("erasure", "heralded erasure", "$p_\\mathrm{erase}$")]


def load(prefix, noise):
    with open(os.path.join(HERE, f"{prefix}_{noise}_threshold_vs_delta_dense.csv")) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        xs = [float(r[c]) for c in ("cross_12x18", "cross_18x24") if r[c]]
        fallback = any(r.get(c, "").startswith("fallback") for c in ("source_12x18", "source_18x24"))
        out.append((float(r["delta"]), xs, fallback))
    return sorted(out)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (noise, title, xlab) in zip(axes, PANELS):
    print(f"\n{noise}: threshold, mean of 12x18 and 18x24 crossings (%)")
    for label, prefix, colour, ls, mk in SERIES:
        rows = load(prefix, noise)
        full = [(d, xs, fb) for d, xs, fb in rows if len(xs) == 2]
        d = np.array([r[0] for r in full])
        lo, mid, hi = (np.array([f(r[1]) for r in full]) * 100 for f in (min, np.mean, max))
        ax.errorbar(d, mid, yerr=[mid - lo, hi - mid], color=colour, ls=ls, lw=1.8, marker=mk,
                    ms=6, capsize=3, label=label, zorder=3)
        fb = np.array([r[2] for r in full])
        if fb.any():
            ax.plot(d[fb], mid[fb], ls="none", marker=mk, ms=6, mfc="white", mec=colour, mew=1.6, zorder=4)
        missing = [r[0] for r in rows if len(r[1]) < 2]
        print(f"   {label:<26}" + "  ".join(f"{x:g}:{m:.2f}" for x, m in zip(d, mid)))
        if missing:
            print(f"   {'':<26}not plotted (a crossing missing): δ = {', '.join(f'{x:g}' for x in missing)}")
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("δ   (0: XX slots only,  1: ZZ slots only)")
    ax.set_ylabel(f"threshold {xlab}  (%)")
    ax.set_xticks(np.linspace(0, 1, 9))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.22, linewidth=.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].legend(fontsize=9, frameon=False, loc="upper center")
fig.suptitle("Threshold vs slot bias δ: unmodified hexagon code vs HexagonMOD2 (no H-fusion)\n"
             "point = mean of the 12×18 and 18×24 crossings, bar = their span; "
             "open marker = 18×24 crossing from the fallback pass", fontsize=10.5)
fig.tight_layout()
out = os.path.join(HERE, "hex_vs_mod2_threshold_vs_delta_dense.png")
fig.savefig(out, dpi=160)
print(f"\nsaved {out}")
