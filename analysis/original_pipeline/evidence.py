"""Evidence registry: resolves every quantity used by tables, figures and manuscript text from the audited artifacts and records it in the ledger.
Token syntax inside text/table templates:  [[AGG method N m metric stat fmt [scale]]]  [[PM method N m metric fmt [scale]]]  [[MI method N m metric fmt [scale]]]
[[FF twin N m metric stat fmt]]  [[ALLMAX metric fmt]]  [[ALLMIN metric fmt]]  [[SUM metric fmt]]  [[CONST name]]  ('-' means m = None)."""
import os, re, json, hashlib
import numpy as np, pandas as pd
B = "/mnt/user-data/outputs/bima_ktsvm_study/"; P11 = B + "phase11/"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
W = pd.read_csv(P11 + "phase11b_results_wide.csv"); A = pd.read_csv(P11 + "phase11b_aggregates.csv"); Fdf = pd.read_csv(P11 + "phase11b_function_fidelity_long.csv")
W["hvp_total"] = W["plus_hvp_count"] + W["minus_hvp_count"]; W["T_solve_total"] = W["T_solve_plus"] + W["T_solve_minus"]; W["r_inf_max"] = W[["plus_r_inf", "minus_r_inf"]].max(axis=1); W["gap_rel_max"] = W[["plus_gap_rel", "minus_gap_rel"]].max(axis=1)
W["cg_cap_total"] = W["plus_cg_cap_events"] + W["minus_cg_cap_events"]; W["zero_curv_total"] = W["plus_zero_curvature_descent"] + W["plus_zero_curvature_nondescent"] + W["minus_zero_curvature_descent"] + W["minus_zero_curvature_nondescent"]

# ---- revision (camera-ready): per-cell extras read from the terminal result records (NYS diagnostics) and derived quantities
_SEL = json.load(open(P11 + "phase11a_selected_cells.json"))["cells"]; RXD = {}
for _c in _SEL:
    _r = json.load(open(_c["result_checked_path"])); _tw = _r["twins"]; _sp = _r.get("spectrum") or {}
    RXD[_c["cell_id"]] = {"cond_A_max": max(_tw[t].get("cond_A", float("nan")) for t in ("plus", "minus")) if _r["cell"]["method"] == "NYS" else float("nan"),
                          "cond_B": _tw["plus"].get("cond_B", float("nan")), "lam_min_ret": _sp.get("min_retained_abs_lambda", float("nan")),
                          "verif_fail": sum(_tw[t].get("refresh_events", {}).get("verification_failures", 0) for t in ("plus", "minus")),
                          "nonfinite": sum(0 if _tw[t].get("finite_reduced_parameters", True) else 1 for t in ("plus", "minus"))}
W["ms_per_hvp"] = 1000.0 * W["T_solve_total"] / W["hvp_total"]
for _k in ("cond_A_max", "cond_B", "lam_min_ret", "verif_fail", "nonfinite"): W[_k] = W["cell_id"].map(lambda c: RXD[c][_k])
def _filt(f):
    q = W if f == "ALL" else None
    if q is None:
        p_ = f.split(":"); q = W[W.method == p_[0]]
        if len(p_) > 1: q = q[q.N == int(p_[1])]
        if len(p_) > 2: q = q[q.m == int(p_[2])]
    return q
def _pairvals(me, N, m, met): q = W[(W.method == me) & (W.N == N)]; q = q[q.m.isna()] if m is None else q[q.m == m]; return dict(zip(q.seed, q[met]))
SRC = {"agg": ("phase11b_aggregates.csv", sha(P11 + "phase11b_aggregates.csv")), "wide": ("phase11b_results_wide.csv", sha(P11 + "phase11b_results_wide.csv")), "ff": ("phase11b_function_fidelity_long.csv", sha(P11 + "phase11b_function_fidelity_long.csv"))}
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
def fmt(x, spec):
    if spec.startswith("f"): return f"{x:,.{int(spec[1:])}f}"
    if spec.startswith("s"): m, e = f"{x:.{int(spec[1:])}e}".split("e"); return f"{m} × 10{str(int(e)).translate(SUP)}"
    if spec == "i": return f"{round(x):,}"
    raise ValueError(spec)
LEDGER = []
LIT = []
def cells_of(method, N, m): q = W[(W.method == method) & (W.N == N)]; q = q[q.m.isna()] if m is None else q[q.m == m]; return q.sort_values("seed")["cell_id"].tolist()
def _agg(method, N, m, metric, stat):
    q = A[(A.method == method) & (A.N == N) & (A.metric == metric)]; q = q[q.m.isna()] if m is None else q[q.m == m]; assert len(q) == 1, (method, N, m, metric); return float(q[stat].iloc[0])
def _m(x): return None if x == "-" else int(x)
def _uniq(N):
    q = pd.read_csv(P11 + "sample_composition.csv"); q = q[q.N == N]["unique_predictor_vectors"]; return f"{int(q.min()):,}–{int(q.max()):,}"
DGJ = json.load(open(P11 + "phase11_operator_diagnostics.json"))
def _alloc(N):
    t = open(B + "phase9b/phase9b_final_protocol.md", encoding="utf8").read(); k = {2000: "2k", 4000: "4k", 8000: "8k"}[N]; a, b = re.findall(r"\b" + k + r" (\d+)/(\d+)", t)[0]; return f"{int(a):,}/{int(b):,}"
def _probe_alloc():
    t = open(B + "phase9b/probe_generator.py", encoding="utf8").read(); a = re.findall(r"EVAL_ALLOC = \{1: (\d+), 0: (\d+)\}", t)[0]; return f"{int(a[0]):,}/{int(a[1]):,}"
def _const(name):
    g0 = json.load(open(B + "g0/g0_raw_data_validation.json")); sch = json.load(open(B + "g1/feature_schema_v2.json")); fp = json.load(open(B + "phase9/phase9_validated_implementation_manifest.json"))["frozen_parameters"]
    pr = json.load(open(B + "phase9b/phase9b_eval_probe_ids.json")); pi = json.load(open(B + "phase9b/phase9b_kernel_pair_manifests/index.json"))["manifests"][0]; r0 = json.load(open(json.load(open(P11 + "phase11a_selected_cells.json"))["cells"][0]["result_checked_path"])); env = json.load(open(B + "phase10/phase10_prelaunch_audit.json"))["C_environment"]
    hw = json.load(open(B + "phase9/phase9_hardware_audit.json")); g3c = json.load(open(B + "g3r2/g3r2_pre_registered_config.json")); mf = json.load(open(B + "phase9/phase9_validated_implementation_manifest.json"))["frozen_parameters"]
    import re as _re
    _tau = float(_re.findall(r"([0-9.e\-]+)\*max", mf["tau_W"])[0]); _tau_s = fmt(_tau, "s0").replace("1 × ", "")
    _sel = json.load(open(P11 + "phase11a_selected_cells.json"))["cells"]; _spec = json.load(open(_sel[0]["canonical_attempt_dir"] + "/spec.json"))
    tab = {"train_sha12": (g0["partitions"]["train"]["sha256"][:12], "g0/g0_raw_data_validation.json"), "test_sha12": (g0["partitions"]["test"]["sha256"][:12], "g0/g0_raw_data_validation.json"), "time_limit_h": (_spec["time_limit_s"] / 3600.0, "spec.json of the canonical attempt"), "gamma": (mf["gamma"], "phase9/phase9_validated_implementation_manifest.json"), "theta": (f"{mf['theta']:g}".replace("-", "−"), "phase9/phase9_validated_implementation_manifest.json"), "c": (f"{mf['c']:g}", "phase9/phase9_validated_implementation_manifest.json"), "d": (f"{mf['d']:g}", "phase9/phase9_validated_implementation_manifest.json"),
           "tau": (_tau_s + "·max<sub>i</sub>|λ<sub>i</sub>(W)|", "phase9/phase9_validated_implementation_manifest.json"), "dup_train": (g0["partitions"]["train"]["duplicate_predictor_vectors_42_columns"], "g0/g0_raw_data_validation.json"), "dup_train_pct": (100.0 * g0["partitions"]["train"]["duplicate_predictor_vectors_42_columns"] / g0["partitions"]["train"]["rows"], "g0/g0_raw_data_validation.json (derived)"),
           "hvp_tol": (g3c["tolerances"]["hvp_rel"], "g3r2/g3r2_pre_registered_config.json"), "mem_gb": (int(hw["ram"]["MemTotal"].split()[0]) * 1024 / 1e9, "phase9/phase9_hardware_audit.json (derived)"), "cpu_model": (hw["cpu_model"], "phase9/phase9_hardware_audit.json"), "cores": (hw["logical_cores_os_cpu_count"], "phase9/phase9_hardware_audit.json"),
           "tol_r": (float(_re.findall(r"r_inf <= ([0-9e.\-]+)", mf["gpcg_thresholds"])[0]), "phase9/phase9_validated_implementation_manifest.json"), "tol_gap": (float(_re.findall(r"gap_rel <= ([0-9e.\-]+)", mf["gpcg_thresholds"])[0]), "phase9/phase9_validated_implementation_manifest.json"),
           "n_seeds": (int(W["seed"].nunique()), "phase11b_results_wide.csv"), "n_exact": (int((W.method == "EXACT").sum()), "phase11b_results_wide.csv"), "n_nys": (int((W.method == "NYS").sum()), "phase11b_results_wide.csv"),
           "train_rows": (g0["partitions"]["train"]["rows"], "g0/g0_raw_data_validation.json"), "test_rows": (g0["partitions"]["test"]["rows"], "g0/g0_raw_data_validation.json"), "n_predictors": (g0["partitions"]["train"]["predictor_columns"]["n_predictors"], "g0/g0_raw_data_validation.json"),
           "n_numerical": (sch["n_raw_numerical_features"], "g1/feature_schema_v2.json"), "n_categorical": (len(sch["categorical_variables"]), "g1/feature_schema_v2.json"), "dim": (sch["total_dimension"], "g1/feature_schema_v2.json"), "hvp_budget": (fp["gpcg_hvp_budget_per_twin"], "phase9/phase9_validated_implementation_manifest.json"),
           "n_pairs": (pi["n_pairs"], "phase9b/phase9b_kernel_pair_manifests/index.json"), "n_probe": (pr["size"], "phase9b/phase9b_eval_probe_ids.json"), "python": (r0["environment"]["python"], "phase10 result.checked.json (environment)"), "numpy": (r0["environment"]["numpy"], "phase10 result.checked.json (environment)"), "scipy": (env["scipy"], "phase10/phase10_prelaunch_audit.json"),
           "n_cells": (len(json.load(open(P11 + "phase11a_selected_cells.json"))["cells"]), "phase11/phase11a_selected_cells.json"),
           "hvp_val_max": (max(r_["max_relative_error"] for r_ in json.load(open(B + "g3r2/g3r2_hvp_equivalence.json"))["rows"]), "g3r2/g3r2_hvp_equivalence.json"),
           "hvp_val_Ns": (" , ".join(str(n) for n in sorted({r_["N"] for r_ in json.load(open(B + "g3r2/g3r2_hvp_equivalence.json"))["rows"]})).replace(" , ", ", ").rsplit(", ", 1)[0] + " and " + str(max(r_["N"] for r_ in json.load(open(B + "g3r2/g3r2_hvp_equivalence.json"))["rows"])), "g3r2/g3r2_hvp_equivalence.json"),
           "hvp_val_vectors": (json.load(open(B + "g3r2/g3r2_hvp_equivalence.json"))["rows"][0]["n_vectors"], "g3r2/g3r2_hvp_equivalence.json"),
           "fl_val_Ns": (" and ".join(str(n) for n in sorted({c_["N"] for c_ in json.load(open(B + "g3r2/g3r2_full_landmark_exact_equivalence.json"))["comparisons"]})), "g3r2/g3r2_full_landmark_exact_equivalence.json"),
           "fl_out_maxdiff": (max(c_["max_abs_raw_output_diff"] for c_ in json.load(open(B + "g3r2/g3r2_full_landmark_exact_equivalence.json"))["comparisons"]), "g3r2/g3r2_full_landmark_exact_equivalence.json"),
           "fl_out_maxabs": (max(c_["max_abs_exact_output"] for c_ in json.load(open(B + "g3r2/g3r2_full_landmark_exact_equivalence.json"))["comparisons"]), "g3r2/g3r2_full_landmark_exact_equivalence.json"),
           "blas": (" ".join(json.load(open(B + "phase9b/phase9b_environment.json"))["blas"]).replace("scipy-openblas", "OpenBLAS (scipy-openblas)"), "phase9b/phase9b_environment.json"),
           "os_kernel": ("Linux " + json.load(open(B + "phase9b/phase9b_environment.json"))["platform"].split("-")[1], "phase9b/phase9b_environment.json"),
           "scipy_env": (json.load(open(B + "phase9b/phase9b_environment.json"))["scipy"], "phase9b/phase9b_environment.json"),
           "alloc_2000": (_alloc(2000), "phase9b/phase9b_final_protocol.md (largest-remainder allocation)"), "alloc_4000": (_alloc(4000), "phase9b/phase9b_final_protocol.md"), "alloc_8000": (_alloc(8000), "phase9b/phase9b_final_protocol.md"),
           "probe_alloc": (_probe_alloc(), "phase9b/probe_generator.py EVAL_ALLOC"), "ram_frac_pct": (int(round(100 * float(re.findall(r"<= ([0-9.]+) x CURRENT MemAvailable", json.load(open(B + "phase9b/phase9b_resource_policy.json"))["memory_rule"])[0]))), "phase9b/phase9b_resource_policy.json"), "gamma_scan_n": (json.load(open("/mnt/project/selected_kernel_config.json"))["kernel_scan_subset_size"], "project file selected_kernel_config.json"), "uniq_2000": (_uniq(2000), "phase11/sample_composition.csv"), "uniq_4000": (_uniq(4000), "phase11/sample_composition.csv"), "uniq_8000": (_uniq(8000), "phase11/sample_composition.csv"), "pilot_dim": (int(re.findall(r'expected_dimension"\] = (\d+)', open(B + "g1/g1_run.py").read())[0]), "g1/g1_run.py (legacy encoding dimension)")}
    return tab[name]
def resolve(tok, location, sentence=""):
    parts = tok.split(); k = parts[0]; srcs = ["agg"]; cells = []; num = None
    if k == "AGG": _, me, N, m, met, st, f = parts[:7]; sc = float(parts[7]) if len(parts) > 7 else 1.0; num = _agg(me, int(N), _m(m), met, st) / sc; out = fmt(num, f); cells = cells_of(me, int(N), _m(m)); desc = f"{st} of {met} ({me}, N={N}, m={m}) / {sc:g}"
    elif k == "PM": _, me, N, m, met, f = parts[:6]; sc = float(parts[6]) if len(parts) > 6 else 1.0; a, s = _agg(me, int(N), _m(m), met, "mean") / sc, _agg(me, int(N), _m(m), met, "sd") / sc; num = a; out = f"{fmt(a, f)} ± {fmt(s, f)}"; cells = cells_of(me, int(N), _m(m)); desc = f"mean ± SD of {met} ({me}, N={N}, m={m}) / {sc:g}"
    elif k == "MI": _, me, N, m, met, f = parts[:6]; sc = float(parts[6]) if len(parts) > 6 else 1.0; a, q1, q3 = [_agg(me, int(N), _m(m), met, s) / sc for s in ("median", "q1", "q3")]; num = a; out = f"{fmt(a, f)} [{fmt(q1, f)}–{fmt(q3, f)}]"; cells = cells_of(me, int(N), _m(m)); desc = f"median [IQR] of {met} ({me}, N={N}, m={m}) / {sc:g}"
    elif k == "FF": _, tw, N, m, met, st, f = parts[:7]; num = _agg("NYS_vs_EXACT_" + tw, int(N), _m(m), met, st); out = fmt(num, f); cells = Fdf[(Fdf.N == int(N)) & (Fdf.m == int(m)) & (Fdf.twin == tw)]["nystrom_cell"].tolist(); srcs = ["agg", "ff"]; desc = f"{st} of {met} (Nystrom vs exact, twin {tw}, N={N}, m={m})"
    elif k in ("ALLMAX", "ALLMIN", "SUM"): _, met, f = parts[:3]; v = W[met]; num = float(v.max() if k == "ALLMAX" else v.min() if k == "ALLMIN" else v.sum()); out = fmt(num, f); cells = W["cell_id"].tolist(); srcs = ["wide"]; desc = f"{k.lower()} of {met} over the 40 cells"
    elif k == "CONST": v, s = _const(parts[1]); num = v if isinstance(v, (int, float)) else None; out = f"{v:,}" if isinstance(v, int) else str(v); srcs = []; desc = f"constant {parts[1]} from {s}"; cells = []
    elif k == "CONSTF": v, s_ = _const(parts[1]); num = float(v); out = fmt(num, parts[2]); srcs = []; desc = f"constant {parts[1]} from {s_}"; cells = []
    elif k == "MONO":
        _, met, me, m, st = parts[:5]; sub = A[(A.method == me) & (A.metric == met)]; sub = sub[sub.m.isna()] if _m(m) is None else sub[sub.m == _m(m)]; Ns = sorted(sub.N.unique()); vals = [_agg(me, int(N), _m(m), met, st) for N in Ns]
        out = "increased" if all(b > a for a, b in zip(vals, vals[1:])) else ("decreased" if all(b < a for a, b in zip(vals, vals[1:])) else "changed non-monotonically"); cells = sum([cells_of(me, int(N), _m(m)) for N in Ns], []); desc = f"trend of {st} {met} over N={Ns} ({me}, m={m}): {out}"
    elif k == "CMPCOUNT":
        _, met, st, me1, m1, me2, m2, Nl = parts[:8]; Ns = [int(x) for x in Nl.split(",")]; kk = sum(_agg(me1, N, _m(m1), met, st) < _agg(me2, N, _m(m2), met, st) for N in Ns); num = float(kk); out = f"{kk} of {len(Ns)}"; cells = sum([cells_of(me1, N, _m(m1)) + cells_of(me2, N, _m(m2)) for N in Ns], []); desc = f"count of N in {Ns} with {st} {met} ({me1},m={m1}) < ({me2},m={m2})"
    elif k == "FFRANGE": _, met, st, f = parts[:4]; v = A[A.method.str.startswith("NYS_vs_EXACT") & (A.metric == met)][st]; out = f"{fmt(float(v.min()), f)}–{fmt(float(v.max()), f)}"; num = float(v.min()); srcs = ["agg", "ff"]; cells = Fdf["nystrom_cell"].tolist(); desc = f"range of {st} {met} over the 8 (twin, N, m) configurations"
    elif k == "FFCMP":
        _, met, st, ma, mb = parts[:5]; kk = n_ = 0
        for tw in ("plus", "minus"):
            for N in (2000, 4000): n_ += 1; kk += _agg("NYS_vs_EXACT_" + tw, N, int(ma), met, st) > _agg("NYS_vs_EXACT_" + tw, N, int(mb), met, st)
        num = float(kk); out = f"{kk} of {n_}"; srcs = ["agg", "ff"]; cells = Fdf["nystrom_cell"].tolist(); desc = f"count of (twin, N) with {st} {met} at m={ma} > m={mb}"
    elif k == "ISMAX":
        _, met, me, m, st, N = parts[:6]; sub = A[(A.method == me) & (A.metric == met)]; sub = sub[sub.m.isna()] if _m(m) is None else sub[sub.m == _m(m)]; vals = {int(n): _agg(me, int(n), _m(m), met, st) for n in sub.N.unique()}
        out = "the highest" if vals[int(N)] == max(vals.values()) else "not the highest"; cells = sum([cells_of(me, n, _m(m)) for n in vals], []); desc = f"is N={N} the maximum of {st} {met} over sizes ({me}, m={m}): {out}"
    elif k == "NYSARR":
        q = W[W.method == "NYS"]; kk = int((q["max_array_bytes"] < q["N"] * q["N"] * 8).sum()); num = float(kk); out = f"{kk} of {len(q)}"; srcs = ["wide"]; cells = q["cell_id"].tolist(); desc = "Nystrom cells whose largest tracked array is smaller than an NxN float64 matrix"
    elif k == "MINSOLVEFRAC":
        q = W[W.method == "NYS"]; v = float((q["T_solve_total"] / q["T_train"] * 100).min()); num = v; out = fmt(v, parts[1]); srcs = ["wide"]; cells = q["cell_id"].tolist(); desc = "minimum over Nystrom cells of 100*(T_solve_plus+T_solve_minus)/T_train"
    elif k == "GT":
        _, ma, mb, me, m, Nl = parts[:6]; Ns = [int(x) for x in Nl.split(",")]; kk = sum(_agg(me, n, _m(m), ma, "median") > _agg(me, n, _m(m), mb, "median") for n in Ns); num = float(kk); out = f"{kk} of {len(Ns)}"; cells = sum([cells_of(me, n, _m(m)) for n in Ns], []); desc = f"count of N in {Ns} with median {ma} > median {mb} ({me}, m={m})"
    elif k == "RX":
        _, met, ag, f, fl = parts[:5]; q = _filt(fl); v = q[met].astype(float); num = float({"max": v.max(), "min": v.min(), "median": v.median(), "sum": v.sum()}[ag]); out = fmt(num, f); srcs = ["wide"]; cells = q["cell_id"].tolist(); desc = f"{ag} of {met} over {fl} cells (terminal result records)"
    elif k == "PAIRX":
        _, met, m, Nl = parts[:4]; kk = n_ = 0
        for N in [int(x) for x in Nl.split(",")]:
            a, b = _pairvals("NYS", N, int(m), met), _pairvals("EXACT", N, None, met)
            for sd in sorted(b): n_ += 1; kk += a[sd] < b[sd]
        num = float(kk); out = f"{kk} of {n_}"; srcs = ["wide"]; cells = W[(W.N.isin([int(x) for x in Nl.split(",")])) & ((W.method == "EXACT") | (W.m == int(m)))]["cell_id"].tolist(); desc = f"paired (N, seed) count with NYS m={m} {met} < EXACT, N in {Nl}"
    elif k == "PAIRK":
        _, met, ma, mb, Nl = parts[:5]; kk = n_ = 0
        for N in [int(x) for x in Nl.split(",")]:
            a, b = _pairvals("NYS", N, int(ma), met), _pairvals("NYS", N, int(mb), met)
            for sd in sorted(b): n_ += 1; kk += a[sd] < b[sd]
        num = float(kk); out = f"{kk} of {n_}"; srcs = ["wide"]; cells = W[(W.method == "NYS") & W.N.isin([int(x) for x in Nl.split(",")])]["cell_id"].tolist(); desc = f"paired (N, seed) count with {met} at m={ma} < m={mb}"
    elif k == "PAIRF":
        _, met, ma, mb, dr = parts[:5]; kk = n_ = 0
        for tw in ("plus", "minus"):
            for N in (2000, 4000):
                qa = Fdf[(Fdf.twin == tw) & (Fdf.N == N) & (Fdf.m == int(ma))].set_index("seed")[met]; qb = Fdf[(Fdf.twin == tw) & (Fdf.N == N) & (Fdf.m == int(mb))].set_index("seed")[met]
                for sd in sorted(qb.index): n_ += 1; kk += (qa[sd] < qb[sd]) if dr == "lt" else (qa[sd] > qb[sd])
        num = float(kk); out = f"{kk} of {n_}"; srcs = ["ff"]; cells = Fdf["nystrom_cell"].tolist(); desc = f"paired (twin, N, seed) count with {met} at m={ma} {dr} m={mb}"
    elif k == "FFSD":
        _, f = parts[:2]; v = Fdf["std_exact_population"].astype(float); num = float(v.min()); out = f"{fmt(float(v.min()), f)}–{fmt(float(v.max()), f)}"; srcs = ["ff"]; cells = Fdf["exact_cell"].tolist(); desc = "range of the population SD of the exact raw functions on the probe"
    elif k == "DG":
        _, part, met, ag, f = parts[:5]; v = np.array([x[met] for x in DGJ["cells" if part == "cells" else "qr_check"]], float); num = float({"max": v.max(), "min": v.min()}[ag]); out = fmt(num, f); srcs = []; cells = [x["cell_id"] for x in DGJ["cells"]]; desc = f"{ag} of {met} in phase11_operator_diagnostics.json ({part})"
    elif k == "DGCNT":
        _, met = parts[:2]; v = [bool(x[met]) for x in DGJ["cells"]]; num = float(sum(v)); out = f"{sum(v)} of {len(v)}"; srcs = []; cells = [x["cell_id"] for x in DGJ["cells"]]; desc = f"count of cells with {met} in phase11_operator_diagnostics.json"
    else: raise ValueError(tok)
    LEDGER.append({"manuscript_location": location, "claim_id": f"C{len(LEDGER)+1:04d}", "claim_text": sentence[:300], "claim_type": "quantitative" if num is not None else "configuration", "numeric_value_if_any": num, "displayed_value": out, "token": tok, "source_artifact": ";".join(SRC[s][0] for s in srcs) if srcs else ("phase11_operator_diagnostics.json" if k in ("DG", "DGCNT") else _const(parts[1])[1]), "source_hash": ";".join(SRC[s][1] for s in srcs) if srcs else (sha(P11 + "phase11_operator_diagnostics.json") if k in ("DG", "DGCNT") else "see artifact"), "source_cells": ";".join(cells), "derivation_script": "phase11b_extract.py -> evidence.py (" + desc + ")", "verification_status": "PENDING"})
    return out
def fill(text, location):
    LIT.append({"location": location, "literal_text": re.sub(r"\[\[.+?\]\]", "§", text)})
    return re.sub(r"\[\[(.+?)\]\]", lambda mo: resolve(mo.group(1), location, re.sub(r"\[\[.+?\]\]", "…", text)), text)
