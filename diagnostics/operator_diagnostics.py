#!/usr/bin/env python3
"""Portable copy (paths only changed) of the script that produced phase11_operator_diagnostics.json. Usage: python diagnostics/operator_diagnostics.py --data-dir <folder with the official CSV> [--limit k]
Post-hoc operator diagnostics requested by reviewer comment 2 (no KTSVM training, no solver run).
For each of the 30 Nystrom cells: rebuild X with the frozen preprocessing from the frozen manifests, verify its SHA-256 against the cell record, rebuild C and the
signed Nystrom factor with the frozen code, verify rank/sign counts/retained-eigenvalue range against the record, and compute kappa_2(Phi) from the singular values
of Phi (without forming Phi^T Phi). For seed 101 (one cell per configuration) compare the production Cholesky-based HVP with an HVP built from a thin QR
factorization of P = blockdiag(Phi, 1) on 10 random vectors per twin."""
import os, sys, json, hashlib, datetime
import numpy as np, pandas as pd
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", required=True, help="folder with UNSW_NB15_training-set.csv"); ap.add_argument("--limit", type=int, default=None, help="only the first k Nystrom cells"); args = ap.parse_args()
for p_ in ("g1", "g2", "g2r", "g2r3", "g3r1", "phase9b"): sys.path.insert(0, str(ROOT / "src" / p_))
import g2r_common as G, exact_ktsvm as ek, preprocessing as pp
from signed_nystrom import signed_nystrom
from nystrom_ktsvm_matrixfree_spd import TwinOperatorSPD
from scipy.linalg import cho_factor, cho_solve
sha_arr = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
sel = [{"cell_id": c, "canonical_attempt_dir": str(ROOT / "results" / c), "result_checked_path": str(ROOT / "results" / c / "result.checked.json")} for c in sorted(os.listdir(ROOT / "results")) if c.startswith("NYS")][: args.limit]
train_df = pp.read_partition(os.path.join(args.data_dir, "UNSW_NB15_training-set.csv"))
PRED = G.NUM + ["proto", "service", "state"]
rows, qr_rows = [], []
for c in sel:
    spec = json.load(open(c["canonical_attempt_dir"] + "/spec.json")); rec = json.load(open(c["result_checked_path"])); m = int(spec["m"]); tag = f"s{spec['seed']}_N{int(spec['N']):06d}.npy"
    P = {"sample": str(ROOT / "manifests" / "sampling" / tag), "landmarks": str(ROOT / "manifests" / "landmarks" / ("lm_" + tag))}
    ids = np.load(P["sample"]); sdf = train_df[train_df["id"].isin(ids)].sort_values("id").reset_index(drop=True); assert (sdf["id"].to_numpy() == ids).all()
    pre = pp.PreprocessorV2(G.NUM, G.VOCAB).fit_scaler(sdf); X = pre.transform(sdf); t = np.where(sdf["label"].to_numpy() == 1, 1, -1)
    lm = np.load(P["landmarks"])[:m]; Lpos = np.searchsorted(ids, lm); assert (ids[Lpos] == lm).all()
    ny = signed_nystrom(ek.sigmoid_gram(X, X[Lpos]), Lpos); lam = ny["eigenvalues_W"][ny["retained"]]; sp = rec["spectrum"]
    s = np.linalg.svd(ny["Phi"], compute_uv=False)
    row = {"cell_id": c["cell_id"], "N": int(spec["N"]), "m": m, "seed": int(spec["seed"]), "X_sha256_matches_record": sha_arr(X) == rec["hashes"]["X_sha256"],
           "rank_matches": ny["r"] == sp["retained_rank_r"], "signs_match": (ny["n_plus"], ny["n_minus"]) == (sp["n_positive_W"], sp["n_negative_W"]),
           "min_ret_lambda_rel_diff": abs(float(np.min(np.abs(lam))) - sp["min_retained_abs_lambda"]) / sp["min_retained_abs_lambda"],
           "max_ret_lambda_rel_diff": abs(float(np.max(np.abs(lam))) - sp["max_retained_abs_lambda"]) / sp["max_retained_abs_lambda"],
           "sigma_max_Phi": float(s[0]), "sigma_min_Phi": float(s[-1]), "kappa2_Phi": float(s[0] / s[-1]), "r": int(ny["r"])}
    if row["seed"] == 101:
        rng = np.random.default_rng(20261109); Phi, J = ny["Phi"], ny["J"]; r = Phi.shape[1]
        Pm = np.zeros((Phi.shape[0] + 1, r + 1)); Pm[:-1, :-1] = Phi; Pm[-1, -1] = 1.0
        Qm, Rm = np.linalg.qr(Pm)                                              # thin QR, P = Q R; no Phi^T Phi formed
        for ell, name in ((1, "plus"), (-1, "minus")):
            op = TwinOperatorSPD(Phi, J, t, ell)
            A_qr = op.d * np.eye(r + 1) + (Rm @ op.Z_l) @ (Rm @ op.Z_l).T; cf = cho_factor(A_qr, lower=True)
            errs = []
            for _ in range(10):
                v = rng.standard_normal(op.n); h1 = op.hvp(v); h2 = op.Z_o.T @ (Rm.T @ cho_solve(cf, Rm @ (op.Z_o @ v)))
                errs.append(float(np.linalg.norm(h1 - h2) / max(1.0, np.linalg.norm(h1))))
            qr_rows.append({"cell_id": c["cell_id"], "N": row["N"], "m": m, "twin": name, "n_vectors": 10, "max_rel_diff_hvp_qr_vs_cholesky": max(errs), "kappa2_R": float(np.linalg.cond(Rm)), "kappa2_A_qr": float(np.linalg.cond(A_qr))})
    rows.append(row); print(c["cell_id"], row["X_sha256_matches_record"], row["rank_matches"], f'{row["kappa2_Phi"]:.3e}', flush=True)
out = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "purpose": "post-hoc operator diagnostics (reviewer comment 2); no training, no solver run",
       "environment_note": "recomputed in the analysis environment, which differs from the execution environment of the 40 cells; reproduction verified by X SHA-256, rank and sign counts",
       "norms": {"kappa2_Phi": "sigma_max(Phi)/sigma_min(Phi) from numpy.linalg.svd(Phi)", "qr_check": "||Hv_chol - Hv_qr||_2 / max(1, ||Hv_chol||_2), 10 standard-normal vectors per twin"},
       "cells": rows, "qr_check": qr_rows}
od = ROOT / "diagnostics" / "reproduced"; od.mkdir(exist_ok=True)
json.dump(out, open(od / "operator_diagnostics.json", "w"), indent=1)
pd.DataFrame(rows).to_csv(od / "operator_diagnostics_cells.csv", index=False); pd.DataFrame(qr_rows).to_csv(od / "operator_diagnostics_qr.csv", index=False)
d = pd.DataFrame(rows); q = pd.DataFrame(qr_rows)
print("X hash matches", int(d.X_sha256_matches_record.sum()), "/", len(d), "| rank", int(d.rank_matches.sum()), "| signs", int(d.signs_match.sum()), "| max lam rel diffs", d.min_ret_lambda_rel_diff.max(), d.max_ret_lambda_rel_diff.max())
print("kappa2(Phi) range", d.kappa2_Phi.min(), d.kappa2_Phi.max()); print(d.groupby(["N", "m"]).kappa2_Phi.median())
print("QR max rel diff", q.max_rel_diff_hvp_qr_vs_cholesky.max()); print(q.to_string())
