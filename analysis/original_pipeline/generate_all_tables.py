#!/usr/bin/env python3
"""Generates every manuscript table from the audited aggregates: display strings, source CSV and provenance JSON (regenerable; no manual entry)."""
import json, datetime, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evidence as EV, pandas as pd
P = EV.P11; now = datetime.datetime.now(datetime.timezone.utc).isoformat(); T = {}
# ---- Table 1: experimental configuration ----
t1 = [("Data", "UNSW-NB15 official training partition ([[CONST train_rows]] rows) for sampling; official testing partition ([[CONST test_rows]] rows) only for the fixed function probe"),
      ("N and seeds", "N = 2,000, 4,000, 8,000 (nested, stratified samples); seeds 101–105"),
      ("Exact KTSVM", "N = 2,000 and 4,000 (five seeds each)"),
      ("Signed Nyström KTSVM", "m = 250 and 1,000 at N = 2,000, 4,000, 8,000 (five seeds each)"),
      ("Kernel and model", "K(x, z) = tanh(γxᵀz + θ), γ = 10/148, θ = −1; c = 1, d = 0.001"),
      ("Solver", "GPCG; stop when r∞ ≤ 10⁻⁶ and gap_rel ≤ 10⁻⁶; HVP budget [[CONST hvp_budget]] per twin"),
      ("Fidelity probes", "[[CONST n_pairs]] fixed training-row pairs per (seed, N); [[CONST n_probe]] fixed test rows"),
      ("Execution", "One process per cell, one BLAS thread; [[CONST n_cells]] terminal cells (10 exact, 30 Nyström)")]
rows = [(a, EV.fill(b, "Table S1")) for a, b in t1]; T["S1"] = {"caption": "Experimental configuration (compact).", "header": ["Item", "Setting"], "rows": [list(r) for r in rows], "widths": [0.22, 0.78]}
pd.DataFrame(rows, columns=T["S1"]["header"]).to_csv(P + "table_S1_source.csv", index=False)
# ---- Table 2: computational resources and convergence ----
cfg = [("EXACT", 2000, "-"), ("EXACT", 4000, "-"), ("NYS", 2000, "250"), ("NYS", 2000, "1000"), ("NYS", 4000, "250"), ("NYS", 4000, "1000"), ("NYS", 8000, "250"), ("NYS", 8000, "1000")]
rows2 = []
for me, N, m in cfg:
    rr = "–" if me == "EXACT" else EV.fill(f"[[AGG {me} {N} {m} retained_rank_r median i]]", "Table 1")
    flt = f"EXACT:{N}" if me == "EXACT" else f"NYS:{N}:{m}"
    rows2.append(["Exact" if me == "EXACT" else "Nyström", f"{N:,}", "–" if m == "-" else f"{int(m):,}", rr, EV.fill(f"[[MI {me} {N} {m} T_train f1]]", "Table 1"), EV.fill(f"[[AGG {me} {N} {m} peak_rss_train_phases_bytes median f0 1000000]]", "Table 1"), EV.fill(f"[[AGG {me} {N} {m} hvp_total median f1 1000]]", "Table 1"), EV.fill(f"[[RX ms_per_hvp median f2 {flt}]]", "Table 1")])
T[1] = {"caption": "Retained rank r, training time (median [IQR] over five paired seeds), and medians of peak RSS in the training phases, total Hessian–vector products of both twins and solver time per HVP (per cell: solver time of both twins divided by their combined HVP count). All 40 cells converged.", "header": ["Method", "N", "m", "r", "Time (s)", "RSS (MB)", "HVP (10³)", "ms/HVP"], "rows": rows2, "widths": [0.13, 0.09, 0.09, 0.07, 0.30, 0.10, 0.11, 0.11]}
pd.DataFrame(rows2, columns=T[1]["header"]).to_csv(P + "table_1_source.csv", index=False)
# ---- Table 3: fidelity ----
rows3 = []
for N in (2000, 4000):
    for m in ("250", "1000"):
        r = [f"{N:,}", f"{int(m):,}", EV.fill(f"[[PM NYS {N} {m} kp_relative_l2 f3]]", "Table 2")]
        if N <= 4000:
            r += [EV.fill(f"[[FF plus {N} {m} f_nrmse mean f2]] ± [[FF plus {N} {m} f_nrmse sd f2]]", "Table 2"), EV.fill(f"[[FF plus {N} {m} f_pearson mean f2]] ± [[FF plus {N} {m} f_pearson sd f2]]", "Table 2"), EV.fill(f"[[FF minus {N} {m} f_nrmse mean f2]] ± [[FF minus {N} {m} f_nrmse sd f2]]", "Table 2"), EV.fill(f"[[FF minus {N} {m} f_pearson mean f2]] ± [[FF minus {N} {m} f_pearson sd f2]]", "Table 2")]
        else: r += ["–"] * 4
        rows3.append(r)
T[2] = {"caption": "Kernel-pair fidelity (relative L2 error, 200,000 pairs) and raw-function fidelity to exact KTSVM (5,000-row probe); mean ± SD over five seeds.", "header": ["N", "m", "Kernel rel. L2", "f₊ NRMSE", "f₊ r", "f₋ NRMSE", "f₋ r"], "rows": rows3, "widths": [0.09, 0.09, 0.2, 0.16, 0.15, 0.16, 0.15]}
pd.DataFrame(rows3, columns=T[2]["header"]).to_csv(P + "table_2_source.csv", index=False)
json.dump(T, open(P + "tables_display.json", "w"), indent=1, ensure_ascii=False)
led = [x for x in EV.LEDGER]; json.dump(led, open(P + "ledger_tables.json", "w"), indent=1, default=str); json.dump(EV.LIT, open(P + "literals_tables.json", "w"), indent=1, ensure_ascii=False)
for n in T:
    cells = sorted({c for x in led if x["manuscript_location"] == f"Table {n}" for c in x["source_cells"].split(";") if c})
    json.dump({"table": n, "caption": T[n]["caption"], "source_csv": f"table_{n}_source.csv", "source_csv_sha256": EV.sha(P + f"table_{n}_source.csv"), "source_cells": cells, "source_artifacts": {k: v for k, v in EV.SRC.items()}, "aggregation_formula": "cell values = mean ± sample SD or median [q1–q3] (numpy linear percentiles) over the five seeds of the configuration, from phase11b_aggregates.csv", "generating_script": "generate_all_tables.py + evidence.py", "timestamp_utc": now}, open(P + f"table_{n}_provenance.json", "w"), indent=1)
for n in T: print(f"Table {n}:", T[n]["header"]); [print("  ", r) for r in T[n]["rows"]]
