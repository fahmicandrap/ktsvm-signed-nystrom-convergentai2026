#!/usr/bin/env python3
"""Generates the two empirical multi-panel figures. Step A writes figure_<n>_source.csv (+ provenance JSON) from the audited wide/function-fidelity tables; Step B draws each figure ONLY from its source CSV
(regenerable: `python3 generate_all_figures.py --from-source <outdir>`). Figure 1: time, peak RSS, HVP count, signed spectrum vs N. Figure 2: kernel-pair error, NRMSE, Pearson. All seeds plotted; no smoothing."""
import json, datetime, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P = os.path.dirname(os.path.abspath(__file__)) + "/"
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
if "--from-source" in sys.argv:
    outdir = sys.argv[sys.argv.index("--from-source") + 1]; os.makedirs(outdir, exist_ok=True)
    for n in (1, 2): plot(n, P + f"figure_{n}_source.csv", outdir + f"/figure_{n}.png")
    print("regenerated from source CSV ->", outdir); sys.exit(0)
import evidence as EV
W, F = EV.W, EV.Fdf; now = datetime.datetime.now(datetime.timezone.utc).isoformat()
CAP = {1: "Training time (log scale), peak RSS in the training phases, total HVP count (log scale) and retained positive/negative eigenvalues of the landmark block versus N; individual seeds and medians. Exact results exist only for N = 2,000 and 4,000.", 2: "Kernel-pair relative L2 error (means) and raw-function NRMSE and Pearson correlation to exact KTSVM (N = 2,000 and 4,000; f+ and f- separately); individual seeds and means."}
S = {1: [], 2: []}; cells = {1: [], 2: []}
for _, r in W.iterrows():
    base = {"N": int(r.N), "m": None if pd.isna(r.m) else int(r.m), "seed": int(r.seed), "source_cell_id": r.cell_id, "source_result_sha256": r.source_result_sha256}
    for panel, col, sc, unit in (("time", "T_train", 1.0, "s"), ("rss", "peak_rss_train_phases_bytes", 1e6, "MB (1e6 bytes)"), ("hvp", "hvp_total", 1.0, "HVP calls (both twins)")): S[1].append({"panel": panel, "series": label(r.m), **base, "value": r[col] / sc, "unit": unit}); cells[1].append(r.cell_id)
    if r.method == "NYS":
        for comp, col in (("positive", "n_positive_W"), ("negative", "n_negative_W")): S[1].append({"panel": "spectrum", "series": comp, **base, "value": r[col], "unit": "retained eigenvalues"}); cells[1].append(r.cell_id)
        S[2].append({"panel": "kernel", "series": label(r.m), **base, "value": r["kp_relative_l2"], "unit": "relative L2 error"}); cells[2].append(r.cell_id)
for _, r in F.iterrows():
    for panel, col in (("NRMSE", "f_nrmse"), ("Pearson correlation", "f_pearson")): S[2].append({"panel": panel, "twin": r.twin, "N": int(r.N), "m": int(r.m), "seed": int(r.seed), "value": r[col], "unit": "dimensionless", "source_cell_id": r.nystrom_cell, "source_exact_cell": r.exact_cell, "source_result_sha256": r.source_nystrom_result_sha256}); cells[2] += [r.nystrom_cell, r.exact_cell]
for n in (1, 2):
    pd.DataFrame(S[n]).to_csv(P + f"figure_{n}_source.csv", index=False); plot(n, P + f"figure_{n}_source.csv", P + f"figure_{n}.png")
    json.dump({"figure": n, "caption": CAP[n], "source_csv": f"figure_{n}_source.csv", "source_csv_sha256": EV.sha(P + f"figure_{n}_source.csv"), "figure_png": f"figure_{n}.png", "source_cells": sorted(set(cells[n])), "aggregation": "every seed is plotted; lines connect per-configuration medians (Fig. 1) or means (kernel panel, Fig. 2); the figure is drawn from the source CSV only", "generating_script": "generate_all_figures.py (regenerate: --from-source <outdir>)", "source_artifacts": {k: v for k, v in EV.SRC.items()}, "exact_series_stops_at_N": 4000, "planning_estimates_used": False, "N16000_used": False, "timestamp_utc": now}, open(P + f"figure_{n}_provenance.json", "w"), indent=1)
print("figures written:", [f"figure_{n}.png" for n in (1, 2)])
