#!/usr/bin/env python3
"""Portable verification of the paper's numbers from this repository alone (no model is trained).
1. checks SHA256SUMS.txt; 2. rebuilds the per-cell quantities from results/*/result.checked.json and the probe outputs;
3. recomputes every quantitative entry of analysis/manuscript_evidence_ledger.csv (tables and text) and compares it with the displayed value;
4. checks the figure source CSVs against the raw records; 5. redraws the figures from their source CSVs into analysis/figures_regenerated/.
Configuration constants (CONST tokens: data sizes, hyperparameters, software versions) are listed with their source artifact in the ledger and are not recomputed here.
Usage: python analysis/reproduce.py"""
import os, sys, json, hashlib, subprocess
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
def fm(x, spec):
    if spec.startswith("f"): return f"{x:,.{int(spec[1:])}f}"
    if spec.startswith("s"): m, e = f"{x:.{int(spec[1:])}e}".split("e"); return f"{m} × 10{str(int(e)).translate(SUP)}"
    if spec == "i": return f"{round(x):,}"
# 1. checksums
bad = []
for line in open(ROOT / "SHA256SUMS.txt", encoding="utf8"):
    h, f = line.rstrip("\n").split("  ", 1)
    if not (ROOT / f).exists() or hashlib.sha256(open(ROOT / f, "rb").read()).hexdigest() != h: bad.append(f)
print(f"[1] SHA256SUMS: {'all files verified' if not bad else 'MISMATCH ' + str(bad[:5])}")
# 2. raw per-cell quantities
sel = [{"cell_id": c, "result_checked_path": str(ROOT / "results" / c / "result.checked.json"), "canonical_attempt_dir": str(ROOT / "results" / c), "result_checked_sha256": hashlib.sha256(open(ROOT / "results" / c / "result.checked.json", "rb").read()).hexdigest()} for c in sorted(os.listdir(ROOT / "results"))]; RAW = []; DIRS = {}
TRAIN = ("T_kernel", "T_factor", "T_operator_", "T_buildH_", "T_solve_")
for c in sel:
    r = json.load(open(c["result_checked_path"])); t = r["timing"]; mem = r["memory"]; tw = r["twins"]; d = {"cell_id": c["cell_id"], "method": r["cell"]["method"], "N": r["cell"]["N"], "m": r["cell"]["m"], "seed": r["cell"]["seed"], "dir": c["canonical_attempt_dir"]}; DIRS[c["cell_id"]] = c["canonical_attempt_dir"]
    d.update({k: t[k] for k in ("T_train", "T_kernel", "T_factor", "T_probe", "T_pairs", "T_total_process", "T_operator_or_buildH")}); d["T_solve_total"] = t["T_solve_plus"] + t["T_solve_minus"]
    d["peak_rss_train_phases_bytes"] = max(v["peak_rss_bytes"] for k, v in mem["phases"].items() if k.startswith(TRAIN)); d["max_array_bytes"] = max(v["max_array_bytes"] for v in mem["major_arrays"].values())
    d["hvp_total"] = tw["plus"]["hvp_count"] + tw["minus"]["hvp_count"]; d["r_inf_max"] = max(tw["plus"]["r_inf"], tw["minus"]["r_inf"]); d["gap_rel_max"] = max(tw["plus"]["gap_rel"], tw["minus"]["gap_rel"])
    d["cg_cap_total"] = tw["plus"]["cg_cap_events"] + tw["minus"]["cg_cap_events"]; d["zero_curv_total"] = sum(tw[x]["zero_curvature_events"][y] for x in ("plus", "minus") for y in ("descent", "nondescent"))
    sp = r.get("spectrum") or {}; kp = r.get("kernel_pair_fidelity") or {}
    d.update({"retained_rank_r": sp.get("retained_rank_r"), "n_positive_W": sp.get("n_positive_W"), "n_negative_W": sp.get("n_negative_W"), "duplicate_landmark_predictor_vectors": sp.get("duplicate_landmark_predictor_vectors"), "kp_relative_l2": kp.get("relative_l2"), "kp_p99_abs_error": kp.get("p99_abs_error"), "kp_max_abs_error": kp.get("max_abs_error"), "kp_rmse_pair": kp.get("rmse_pair"), "status": r["status"], "hvp_budget": json.load(open(d["dir"] + "/spec.json"))["budget"], "n_pairs": kp.get("n_pairs"), "n_probe": r["hashes"]["n_probe"], "env": r["environment"], "res_sha": c["result_checked_sha256"]}); RAW.append(d)
for d, c in zip(RAW, sel):
    r = json.load(open(c["result_checked_path"])); tw = r["twins"]; sp = r.get("spectrum") or {}; t = r["timing"]
    d["ms_per_hvp"] = 1000.0 * (t["T_solve_plus"] + t["T_solve_minus"]) / d["hvp_total"]
    d["cond_A_max"] = max(tw["plus"].get("cond_A", np.nan), tw["minus"].get("cond_A", np.nan)) if d["method"] == "NYS" else np.nan; d["cond_B"] = tw["plus"].get("cond_B", np.nan)
    d["lam_min_ret"] = sp.get("min_retained_abs_lambda", np.nan); d["verif_fail"] = sum(tw[x].get("refresh_events", {}).get("verification_failures", 0) for x in ("plus", "minus")); d["nonfinite"] = sum(0 if tw[x].get("finite_reduced_parameters", True) else 1 for x in ("plus", "minus"))
assert len(RAW) == 40
print(f"[2] Rebuilt {len(RAW)} cell records from results/ (10 exact, 30 signed Nyström)")
def rows(me, N, m): return sorted([d for d in RAW if d["method"] == me and d["N"] == N and d["m"] == m], key=lambda d: d["seed"])
def vec(me, N, m, met): return np.array([d[met] for d in rows(me, N, m)], float)
def st(a, s): return {"mean": a.mean(), "sd": a.std(ddof=1), "median": np.median(a), "q1": np.percentile(a, 25), "q3": np.percentile(a, 75)}[s]
M_ = lambda x: None if x == "-" else int(x)
FF = {}                                              # independent raw-function fidelity (own formulas)
for d in RAW:
    if d["method"] == "NYS" and d["N"] in (2000, 4000):
        ex = rows("EXACT", d["N"], None); ex = [e for e in ex if e["seed"] == d["seed"]][0]
        for tw in ("plus", "minus"):
            fe = np.load(f"{ex['dir']}/probe_f_{tw}.npy").astype(float); fn = np.load(f"{d['dir']}/probe_f_{tw}.npy").astype(float); e = fn - fe; rm = np.sqrt(np.mean(e ** 2))
            FF.setdefault((tw, d["N"], d["m"]), []).append((d["seed"], {"f_rmse": rm, "f_nrmse": rm / np.std(fe), "f_pearson": np.corrcoef(fe, fn)[0, 1], "f_relative_l2": np.linalg.norm(e) / np.linalg.norm(fe), "f_max_abs_deviation": np.abs(e).max(), "f_p99_abs_deviation": np.percentile(np.abs(e), 99)}))
FSD = {}
for d in RAW:
    if d["method"] == "EXACT":
        for tw in ("plus", "minus"): FSD[(tw, d["N"], d["seed"])] = float(np.std(np.load(f"{d['dir']}/probe_f_{tw}.npy").astype(float)))
def ffv(tw, N, m, met): return np.array([v[met] for _, v in sorted(FF[(tw, N, m)], key=lambda x: x[0])], float)
DGJ = json.load(open(ROOT / "diagnostics" / "phase11_operator_diagnostics.json"))
def evaluate(tok):
    p = tok.split(); k = p[0]
    if k == "AGG": _, me, N, m, met, s, f = p[:7]; sc = float(p[7]) if len(p) > 7 else 1.0; v = st(vec(me, int(N), M_(m), met), s) / sc; return fm(v, f), v
    if k == "PM": _, me, N, m, met, f = p[:6]; sc = float(p[6]) if len(p) > 6 else 1.0; a = vec(me, int(N), M_(m), met) / sc; return f"{fm(a.mean(), f)} ± {fm(a.std(ddof=1), f)}", a.mean()
    if k == "MI": _, me, N, m, met, f = p[:6]; sc = float(p[6]) if len(p) > 6 else 1.0; a = vec(me, int(N), M_(m), met) / sc; return f"{fm(np.median(a), f)} [{fm(np.percentile(a, 25), f)}–{fm(np.percentile(a, 75), f)}]", float(np.median(a))
    if k == "FF": _, tw, N, m, met, s, f = p[:7]; v = st(ffv(tw, int(N), int(m), met), s); return fm(v, f), v
    if k in ("ALLMAX", "ALLMIN", "SUM"): _, met, f = p[:3]; a = np.array([d[met] for d in RAW], float); v = a.max() if k == "ALLMAX" else a.min() if k == "ALLMIN" else a.sum(); return fm(v, f), float(v)
    if k == "MONO":
        _, met, me, m, s = p[:5]; Ns = sorted({d["N"] for d in RAW if d["method"] == me and d["m"] == M_(m)}); v = [st(vec(me, n, M_(m), met), s) for n in Ns]; return ("increased" if all(b > a for a, b in zip(v, v[1:])) else "decreased" if all(b < a for a, b in zip(v, v[1:])) else "changed non-monotonically"), None
    if k == "CMPCOUNT": _, met, s, m1, a1, m2, a2, Nl = p[:8]; Ns = [int(x) for x in Nl.split(",")]; kk = sum(st(vec(m1, n, M_(a1), met), s) < st(vec(m2, n, M_(a2), met), s) for n in Ns); return f"{kk} of {len(Ns)}", float(kk)
    if k == "FFRANGE": _, met, s, f = p[:4]; v = [st(ffv(tw, N, m, met), s) for (tw, N, m) in FF]; return f"{fm(min(v), f)}–{fm(max(v), f)}", min(v)
    if k == "FFCMP": _, met, s, ma, mb = p[:5]; kk = n_ = 0; [(None) for _ in ()]; kk = sum(st(ffv(tw, N, int(ma), met), s) > st(ffv(tw, N, int(mb), met), s) for tw in ("plus", "minus") for N in (2000, 4000)); return f"{kk} of 4", float(kk)
    if k == "ISMAX": _, met, me, m, s, N = p[:6]; Ns = sorted({d["N"] for d in RAW if d["method"] == me and d["m"] == M_(m)}); v = {n: st(vec(me, n, M_(m), met), s) for n in Ns}; return ("the highest" if v[int(N)] == max(v.values()) else "not the highest"), None
    if k == "NYSARR": q = [d for d in RAW if d["method"] == "NYS"]; kk = sum(d["max_array_bytes"] < d["N"] * d["N"] * 8 for d in q); return f"{kk} of {len(q)}", float(kk)
    if k == "MINSOLVEFRAC": q = [d for d in RAW if d["method"] == "NYS"]; v = min(d["T_solve_total"] / d["T_train"] * 100 for d in q); return fm(v, p[1]), v
    if k == "GT": _, ma, mb, me, m, Nl = p[:6]; Ns = [int(x) for x in Nl.split(",")]; kk = sum(st(vec(me, n, M_(m), ma), "median") > st(vec(me, n, M_(m), mb), "median") for n in Ns); return f"{kk} of {len(Ns)}", float(kk)
    if k == "RX":
        _, met, ag, f, fl = p[:5]; ff = fl.split(":"); q = [d for d in RAW if fl == "ALL" or (d["method"] == ff[0] and (len(ff) < 2 or d["N"] == int(ff[1])) and (len(ff) < 3 or d["m"] == int(ff[2])))]
        v = np.array([d[met] for d in q], float); v = {"max": np.nanmax(v), "min": np.nanmin(v), "median": np.nanmedian(v), "sum": np.nansum(v)}[ag]; return fm(v, f), float(v)
    if k == "PAIRX": _, met, m, Nl = p[:4]; pr = [(x, y) for n in map(int, Nl.split(",")) for x, y in zip(rows("NYS", n, int(m)), rows("EXACT", n, None))]; assert all(x["seed"] == y["seed"] for x, y in pr); kk = sum(x[met] < y[met] for x, y in pr); return f"{kk} of {len(pr)}", float(kk)
    if k == "PAIRK": _, met, ma, mb, Nl = p[:5]; pr = [(x, y) for n in map(int, Nl.split(",")) for x, y in zip(rows("NYS", n, int(ma)), rows("NYS", n, int(mb)))]; assert all(x["seed"] == y["seed"] for x, y in pr); kk = sum(x[met] < y[met] for x, y in pr); return f"{kk} of {len(pr)}", float(kk)
    if k == "PAIRF":
        _, met, ma, mb, dr = p[:5]; kk = n_ = 0
        for tw in ("plus", "minus"):
            for N in (2000, 4000):
                A_ = dict(FF[(tw, N, int(ma))]); B_ = dict(FF[(tw, N, int(mb))])
                for sd in sorted(B_): n_ += 1; kk += (A_[sd][met] < B_[sd][met]) if dr == "lt" else (A_[sd][met] > B_[sd][met])
        return f"{kk} of {n_}", float(kk)
    if k == "DG":
        _, part, met, ag, f = p[:5]; L_ = DGJ["cells"] if part == "cells" else DGJ["qr_check"]; v = [float(x[met]) for x in L_]; v = max(v) if ag == "max" else min(v); return fm(v, f), v
    if k == "DGCNT": _, met = p[:2]; v = [x for x in DGJ["cells"] if x[met] is True]; return f"{len(v)} of {len(DGJ['cells'])}", float(len(v))
    if k == "FFSD": _, f = p[:2]; v = list(FSD.values()); return f"{fm(min(v), f)}–{fm(max(v), f)}", min(v)
    raise ValueError(tok)

# 3. ledger
led = pd.read_csv(ROOT / "analysis" / "manuscript_evidence_ledger.csv"); n_ok = n_tot = 0; fails = []
for _, r in led.iterrows():
    tok = str(r["token"]); k = tok.split()[0]
    if k in ("CONST", "CONSTF"): continue
    n_tot += 1; disp, _ = evaluate(tok)
    if str(disp) == str(r["displayed_value"]): n_ok += 1
    else: fails.append((r["claim_id"], tok, r["displayed_value"], disp))
print(f"[3] Evidence ledger: {n_ok}/{n_tot} quantitative entries reproduced exactly from results/ and diagnostics/ ({len(led) - n_tot} configuration constants listed with their source)")
for f_ in fails[:10]: print("    MISMATCH", f_)
# 4. figure source CSVs against raw records
f1 = pd.read_csv(ROOT / "analysis" / "figures" / "figure_1_source.csv"); f2 = pd.read_csv(ROOT / "analysis" / "figures" / "figure_2_source.csv"); byid = {d["cell_id"]: d for d in RAW}; worst = 0.0
for _, r in f1.iterrows():
    d = byid[r["source_cell_id"]]; v = {"time": d["T_train"], "rss": d["peak_rss_train_phases_bytes"] / 1e6, "hvp": d["hvp_total"], "spectrum": d["n_positive_W"] if r["series"] == "positive" else d["n_negative_W"]}[r["panel"]]; worst = max(worst, abs(v - r["value"]) / max(1.0, abs(v)))
for _, r in f2.iterrows():
    if r["panel"] == "kernel": v = byid[r["source_cell_id"]]["kp_relative_l2"]
    else: v = dict(FF[(r["twin"], int(r["N"]), int(r["m"]))])[int(r["seed"])]["f_nrmse" if r["panel"] == "NRMSE" else "f_pearson"]
    worst = max(worst, abs(v - r["value"]) / max(1.0, abs(v)))
print(f"[4] Figure source CSVs: {len(f1) + len(f2)} plotted values, max relative difference to raw records = {worst:.1e}")
# 5. redraw figures from source CSVs
subprocess.run([sys.executable, str(ROOT / "analysis" / "plot_figures.py")], check=True)
print("[5] Figures redrawn into analysis/figures_regenerated/ (byte identity with the archived PNGs depends on the matplotlib version)")
sys.exit(0 if (not bad and not fails and worst < 1e-9) else 1)
