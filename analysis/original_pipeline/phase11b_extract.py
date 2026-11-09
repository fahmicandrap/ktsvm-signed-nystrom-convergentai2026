#!/usr/bin/env python3
"""Phase 11B: programmatic extraction of ALL results from the 40 audited terminal artifacts -> canonical long table, wide table, function-fidelity table, descriptive aggregates.
No model is run; no number is typed by hand."""
import os, sys, json, hashlib, datetime
B = "/mnt/user-data/outputs/bima_ktsvm_study/"; OUT = B + "phase11/"; sys.path.insert(0, B + "phase9b")
import numpy as np, pandas as pd
import measurement as M                      # frozen, tested metric definitions (function_fidelity)
sel = json.load(open(OUT + "phase11a_selected_cells.json"))["cells"]; assert len(sel) == 40 and json.load(open(OUT + "phase11a_integrity_audit.json"))["VERDICT"].startswith("COMPACT DATASET READY")
R = {c["cell_id"]: json.load(open(c["result_checked_path"])) for c in sel}; D = {c["cell_id"]: c["canonical_attempt_dir"] for c in sel}; H = {c["cell_id"]: c["result_checked_sha256"] for c in sel}
TRAIN_PREFIX = ("T_kernel", "T_factor", "T_operator_", "T_buildH_", "T_solve_")
def env_id(r): e = r["environment"]; return hashlib.sha256(json.dumps([e["python"], e["numpy"], e["platform"], e["threads"]["env"]], sort_keys=True).encode()).hexdigest()[:12]
by = {(c["method"], c["N"], c["m"], c["seed"]): c["cell_id"] for c in sel}
ffrows, long_rows, wide_rows = [], [], []
for c in sel:
    cid = c["cell_id"]; r = R[cid]; t = r["timing"]; mem = r["memory"]; ph = mem["phases"]; sp = r.get("spectrum") or {}; kp = r.get("kernel_pair_fidelity") or {}
    train_peak = max(v["peak_rss_bytes"] for k, v in ph.items() if k.startswith(TRAIN_PREFIX)); maxarr = max(v["max_array_bytes"] for v in mem["major_arrays"].values()); shp = [v["max_array_shape"] for v in mem["major_arrays"].values() if v["max_array_bytes"] == maxarr][0]
    cell = {"cell_id": cid, "method": c["method"], "N": c["N"], "m": c["m"], "seed": c["seed"], "terminal_status": r["status"], "T_load": t["T_load"], "T_prep": t["T_prep"], "T_kernel": t["T_kernel"], "T_factor": t["T_factor"], "T_operator_or_buildH": t["T_operator_or_buildH"], "T_solve_plus": t["T_solve_plus"], "T_solve_minus": t["T_solve_minus"], "T_train": t["T_train"], "T_probe": t["T_probe"], "T_pairs": t["T_pairs"], "T_total_process": t["T_total_process"],
            "peak_rss_bytes_over_phases": mem["peak_rss_bytes_over_phases"], "peak_rss_train_phases_bytes": train_peak, "ru_maxrss_bytes": mem["ru_maxrss_bytes"], "baseline_rss_bytes": mem["baseline_rss_bytes"], "max_array_bytes": maxarr, "max_array_shape": "x".join(map(str, shp)),
            "retained_rank_r": sp.get("retained_rank_r"), "n_positive_W": sp.get("n_positive_W"), "n_negative_W": sp.get("n_negative_W"), "n_zero_W": sp.get("n_zero_W"), "tau_W": sp.get("tau_W"), "min_retained_abs_lambda": sp.get("min_retained_abs_lambda"), "max_retained_abs_lambda": sp.get("max_retained_abs_lambda"), "duplicate_landmark_predictor_vectors": sp.get("duplicate_landmark_predictor_vectors"),
            "kp_rmse_pair": kp.get("rmse_pair"), "kp_relative_l2": kp.get("relative_l2"), "kp_max_abs_error": kp.get("max_abs_error"), "kp_p99_abs_error": kp.get("p99_abs_error"), "kp_mean_signed_error": kp.get("mean_signed_error_K_minus_Khat"), "environment_id": env_id(r), "source_result_file": c["result_checked_path"], "source_result_sha256": H[cid]}
    wide = dict(cell)
    for tw in ("plus", "minus"):
        s = r["twins"][tw]; oc = s["phase_counts"]; row = dict(cell); row.update({"twin": tw, "dual_dimension": s["dual_dimension"], "twin_status": s["status"], "hvp_count": s["hvp_count"], "hvp_budget": s["hvp_budget"], "hvp_phaseA_trial": s["hvp_by_category"].get("phaseA_trial", 0), "hvp_cg": s["hvp_by_category"].get("cg", 0), "hvp_projected_search": s["hvp_by_category"].get("projected_search", 0), "hvp_verification_refresh": s["hvp_by_category"].get("verification_refresh", 0),
              "r_inf": s["r_inf"], "gap_rel": s["gap_rel"], "dual_objective_D": s["dual_objective_D"], "cg_cap_events": s["cg_cap_events"], "cg_infeasible_steps": s["cg_infeasible_steps"], "zero_curvature_descent": s["zero_curvature_events"]["descent"], "zero_curvature_nondescent": s["zero_curvature_events"]["nondescent"], "refresh_gradient": s["refresh_events"]["gradient"], "refresh_verification": s["refresh_events"]["verification"], "outer_iterations": oc.get("outer_iterations"), "phaseA_exit_same_face": oc.get("phaseA_exit_same_face"), "phaseA_exit_stagnation": oc.get("phaseA_exit_stagnation"),
              "T_solve_twin": t["T_solve_" + tw], "T_operator_or_buildH_twin": t["T_operator_or_buildH_plus" if tw == "plus" else "T_operator_or_buildH_minus"], "cond_B": s.get("cond_B"), "cond_A": s.get("cond_A")})
        for k in ("hvp_count", "r_inf", "gap_rel", "dual_objective_D", "cg_cap_events", "zero_curvature_descent", "zero_curvature_nondescent", "dual_dimension"): wide[f"{tw}_{k}"] = row[k]
        # raw-function fidelity: only Nystrom cells with an exact reference at the same (N, seed): N in {2000, 4000}
        ff = {k: None for k in ("f_rmse", "f_nrmse", "f_pearson", "f_relative_l2", "f_max_abs_deviation", "f_p99_abs_deviation")}
        if c["method"] == "NYS" and c["N"] in (2000, 4000):
            ex = by[("EXACT", c["N"], None, c["seed"])]; fe = np.load(f"{D[ex]}/probe_f_{tw}.npy"); fn = np.load(f"{D[cid]}/probe_f_{tw}.npy"); m_ = M.function_fidelity(fe, fn)
            ff = {"f_rmse": m_["rmse"], "f_nrmse": m_["nrmse"], "f_pearson": m_["pearson"], "f_relative_l2": m_["relative_l2"], "f_max_abs_deviation": m_["max_abs_deviation"], "f_p99_abs_deviation": m_["p99_abs_deviation"]}
            ffrows.append({"nystrom_cell": cid, "exact_cell": ex, "N": c["N"], "m": c["m"], "seed": c["seed"], "twin": tw, **ff, "std_exact_population": float(np.std(fe)), "n_probe": len(fe), "source_exact_result_sha256": H[ex], "source_nystrom_result_sha256": H[cid]})
        row.update(ff); long_rows.append(row)
        for k, v in ff.items(): wide[f"{tw}_{k}"] = v
    wide_rows.append(wide)
L = pd.DataFrame(long_rows); W = pd.DataFrame(wide_rows); F = pd.DataFrame(ffrows)
L.to_csv(OUT + "phase11b_results_long.csv", index=False); W.to_csv(OUT + "phase11b_results_wide.csv", index=False); F.to_csv(OUT + "phase11b_function_fidelity_long.csv", index=False)
# ---- descriptive aggregates (five seeds are computational repetitions; no tests, no CIs) ----
W["hvp_total"] = W["plus_hvp_count"] + W["minus_hvp_count"]; W["T_solve_total"] = W["T_solve_plus"] + W["T_solve_minus"]; W["r_inf_max"] = W[["plus_r_inf", "minus_r_inf"]].max(axis=1); W["gap_rel_max"] = W[["plus_gap_rel", "minus_gap_rel"]].max(axis=1); W["cg_cap_total"] = W["plus_cg_cap_events"] + W["minus_cg_cap_events"]; W["zero_curv_total"] = W["plus_zero_curvature_descent"] + W["plus_zero_curvature_nondescent"] + W["minus_zero_curvature_descent"] + W["minus_zero_curvature_nondescent"]
cellm = ["T_kernel", "T_factor", "T_operator_or_buildH", "T_solve_total", "T_train", "T_probe", "T_pairs", "T_total_process", "peak_rss_bytes_over_phases", "peak_rss_train_phases_bytes", "ru_maxrss_bytes", "max_array_bytes", "hvp_total", "plus_hvp_count", "minus_hvp_count", "r_inf_max", "gap_rel_max", "plus_dual_objective_D", "minus_dual_objective_D", "cg_cap_total", "zero_curv_total", "retained_rank_r", "n_positive_W", "n_negative_W", "n_zero_W", "duplicate_landmark_predictor_vectors", "kp_rmse_pair", "kp_relative_l2", "kp_max_abs_error", "kp_p99_abs_error", "kp_mean_signed_error"]
agg = []
def stats(v):
    v = np.array([x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))], float)
    if len(v) == 0: return None
    return {"n": len(v), "mean": v.mean(), "sd": v.std(ddof=1) if len(v) > 1 else float("nan"), "median": np.median(v), "q1": np.percentile(v, 25), "q3": np.percentile(v, 75), "min": v.min(), "max": v.max(), "values_by_seed": ";".join(f"{x:.10g}" for x in v)}
for (meth, N, m), g in W.groupby(["method", "N", W["m"].fillna(-1)]):
    g = g.sort_values("seed")
    for met in cellm:
        s = stats(g[met].tolist())
        if s: agg.append({"method": meth, "N": N, "m": None if m == -1 else int(m), "metric": met, **s})
for (N, m), g in F.groupby(["N", "m"]):
    for tw in ("plus", "minus"):
        gg = g[g["twin"] == tw].sort_values("seed")
        for met in ("f_rmse", "f_nrmse", "f_pearson", "f_relative_l2", "f_max_abs_deviation", "f_p99_abs_deviation"):
            s = stats(gg[met].tolist())
            if s: agg.append({"method": "NYS_vs_EXACT_" + tw, "N": N, "m": int(m), "metric": met, **s})
A = pd.DataFrame(agg); A.to_csv(OUT + "phase11b_aggregates.csv", index=False)
json.dump({"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "script": "phase11b_extract.py", "rows_long": len(L), "rows_wide": len(W), "rows_function_fidelity": len(F), "aggregates": len(A), "source_cells": [c["cell_id"] for c in sel], "aggregation": "mean, sample SD (ddof=1), median, IQR (numpy linear percentiles) over the five seeds; all seed values retained in values_by_seed; no significance tests"}, open(OUT + "phase11b_extraction_record.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
piv = lambda met, st="median": A[(A.metric == met)].pivot_table(index=["method", "N", "m"], values=st)
print("statuses:", W["terminal_status"].value_counts().to_dict(), "| twin statuses:", L["twin_status"].value_counts().to_dict(), "| env ids:", W["environment_id"].unique().tolist())
for met, st in (("T_train", "median"), ("peak_rss_train_phases_bytes", "median"), ("peak_rss_bytes_over_phases", "median"), ("max_array_bytes", "median"), ("hvp_total", "median"), ("T_solve_total", "median"), ("kp_relative_l2", "mean"), ("kp_rmse_pair", "mean"), ("retained_rank_r", "median")): print("\n", met, st); print(piv(met, st).round(4).to_string())
print("\nmax r_inf over all cells:", W["r_inf_max"].max(), "| max gap_rel:", W["gap_rel_max"].max(), "| cg cap total:", W["cg_cap_total"].sum(), "| zero-curv total:", W["zero_curv_total"].sum())
print("\nfunction fidelity (mean over seeds):"); print(A[A.method.str.startswith("NYS_vs")].pivot_table(index=["method", "N", "m"], columns="metric", values="mean").round(5).to_string())
