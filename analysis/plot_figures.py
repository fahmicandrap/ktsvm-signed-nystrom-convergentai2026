#!/usr/bin/env python3
"""Portable copy of the figure generator (plot function unchanged): redraws Figs. 2 and 3 of the paper from analysis/figures/figure_<n>_source.csv into analysis/figures_regenerated/.
Original docstring: Generates the two empirical multi-panel figures. Step A writes figure_<n>_source.csv (+ provenance JSON) from the audited wide/function-fidelity tables; Step B draws each figure ONLY from its source CSV
(regenerable: `python3 generate_all_figures.py --from-source <outdir>`). Figure 1: time, peak RSS, HVP count, signed spectrum vs N. Figure 2: kernel-pair error, NRMSE, Pearson. All seeds plotted; no smoothing."""
import json, datetime, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures") + "/"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "lines.linewidth": 0.9})
COL = {"Exact": "#000000", "Nyström, m = 250": "#1f77b4", "Nyström, m = 1,000": "#d62728"}; MK = {"Exact": "s", "Nyström, m = 250": "o", "Nyström, m = 1,000": "^"}; SERIES = list(COL)
TC = {"plus": "#1f77b4", "minus": "#ff7f0e"}; SC = {"positive": "#1f77b4", "negative": "#ff7f0e"}
def label(m): return "Exact" if pd.isna(m) else ("Nyström, m = 250" if int(m) == 250 else "Nyström, m = 1,000")
def xlog(ax, xlabel=True):
    ax.set_xscale("log", base=2); ax.set_xticks([2000, 4000, 8000]); ax.set_xticklabels(["2,000", "4,000", "8,000"]); ax.minorticks_off()
    if xlabel: ax.set_xlabel("Training size N")
def tag(ax, t): ax.text(0.02, 0.96, t, transform=ax.transAxes, va="top", fontweight="bold")
def series_panel(ax, d, panel, ylab, use, log=False, agg="median", xlabel=True, legend=False):
    for s in SERIES:
        g = d[(d.panel == panel) & (d.series == s)]
        if len(g) == 0: continue
        ax.scatter(g.N, g.value, s=7, color=COL[s], marker=MK[s], alpha=0.45, linewidths=0); m_ = getattr(g.groupby("N")["value"], agg)(); ax.plot(m_.index, m_.values, color=COL[s], marker=MK[s], ms=3.2, label=s)
    xlog(ax, xlabel); ax.set_ylabel(ylab)
    if log: ax.set_yscale("log")
def plot(n, src, out):
    d = pd.read_csv(src)
    if n == 1:
        fig, ax = plt.subplots(2, 2, figsize=(4.7, 3.05))
        series_panel(ax[0, 0], d, "time", "Training time (s)", None, log=True, xlabel=False); series_panel(ax[0, 1], d, "rss", "Peak RSS (MB)", None, xlabel=False)
        series_panel(ax[1, 0], d, "hvp", "Total HVP count (both twins)", None, log=True)
        for gi, (N, m) in enumerate([(2000, 250), (2000, 1000), (4000, 250), (4000, 1000), (8000, 250), (8000, 1000)]):
            for comp in ("positive", "negative"):
                g = d[(d.panel == "spectrum") & (d.N == N) & (d.m == m) & (d.series == comp)].sort_values("seed"); x = gi + (-0.17 if comp == "positive" else 0.17)
                ax[1, 1].scatter([x + (j - 2) * 0.02 for j in range(len(g))], g.value, s=8, color=SC[comp], alpha=0.5, linewidths=0, label=(comp + " eigenvalues") if gi == 0 else None); ax[1, 1].plot([x - 0.08, x + 0.08], [g.value.median()] * 2, color=SC[comp], lw=1.3)
        ax[1, 1].set_xticks(range(6)); ax[1, 1].set_xticklabels([f"{N // 1000}k\n{m}" for N, m in [(2000, 250), (2000, 1000), (4000, 250), (4000, 1000), (8000, 250), (8000, 1000)]]); ax[1, 1].set_xlabel("N (thousand) / landmarks m"); ax[1, 1].set_ylabel("Retained eigenvalues"); lg = ax[1, 1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9)); [h.set_alpha(1) for h in lg.legend_handles]
        for a, t in zip(ax.flat, "abcd"): tag(a, f"({t})")
        h_, l_ = ax[0, 0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(4.7, 1.85)); groups = [(2000, 250), (2000, 1000), (4000, 250), (4000, 1000)]
        series_panel(ax[0], d, "kernel", "Kernel-pair relative L2 error", None, agg="mean")
        for pi, (pan, ylab) in enumerate((("NRMSE", "NRMSE to exact"), ("Pearson correlation", "Pearson r to exact")), start=1):
            for gi, (N, m) in enumerate(groups):
                for tw in ("plus", "minus"):
                    g = d[(d.panel == pan) & (d.N == N) & (d.m == m) & (d.twin == tw)].sort_values("seed"); x = gi + (-0.17 if tw == "plus" else 0.17)
                    ax[pi].scatter([x + (j - 2) * 0.02 for j in range(len(g))], g.value, s=8, color=TC[tw], alpha=0.5, linewidths=0, label=("f₊" if tw == "plus" else "f₋") if (pi == 1 and gi == 0) else None); ax[pi].plot([x - 0.08, x + 0.08], [g.value.mean()] * 2, color=TC[tw], lw=1.3)
            ax[pi].set_xticks(range(4)); ax[pi].set_xticklabels([f"{N // 1000}k\n{m}" for N, m in groups], fontsize=6.2); ax[pi].set_xlabel("N (thousand) / m"); ax[pi].set_ylabel(ylab)
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], color=COL["Nyström, m = 250"], marker="o", ms=3, lw=0.9), Line2D([], [], color=COL["Nyström, m = 1,000"], marker="^", ms=3, lw=0.9), Line2D([], [], color=TC["plus"], marker="o", ms=3, lw=0), Line2D([], [], color=TC["minus"], marker="o", ms=3, lw=0)]
        fig.legend(hs, ["(a) m = 250", "(a) m = 1,000", "(b,c) f₊", "(b,c) f₋"], frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
        for a, t in zip(ax, "abc"): tag(a, f"({t})")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_regenerated"); os.makedirs(out, exist_ok=True)
for n in (1, 2): plot(n, P + f"figure_{n}_source.csv", os.path.join(out, f"figure_{n}.png"))
print("redrawn:", sorted(os.listdir(out)))
