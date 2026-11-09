"""G2-R common helpers (does not modify the frozen G2 code; imports it read-only).

Problem definition is exactly the frozen exact KTSVM of g2/g2_ktsvm_specification.md (c = 1, d = 0.001).
Only the numerical evaluation of H is made more accurate (iterative refinement), and the same H is used
by every solver and by the validator.
"""
import hashlib, io, json, os, sys, zipfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
sys.path.insert(0, HERE + "../g1/")
sys.path.insert(0, HERE + "../g2/")
import preprocessing as pp  # noqa: E402
import exact_ktsvm as ek     # noqa: E402

C, D = 1.0, 1e-3
CFG = json.load(open(HERE + "g2r_solver_config.json"))
TH = CFG["acceptance_thresholds_unchanged"]


def load_train_sample():
    zb = zipfile.ZipFile("/mnt/user-data/uploads/unsw_nb15_prepared_package.zip").read("unsw_train_sample_2000.csv")
    return pd.read_csv(io.BytesIO(zb), keep_default_na=False), hashlib.sha256(zb).hexdigest()


_vocab_art = json.load(open(HERE + "../g1/categorical_vocabulary_v2.json"))
VOCAB = _vocab_art["vocabulary"]
assert hashlib.sha256(json.dumps(VOCAB, sort_keys=True).encode()).hexdigest() == _vocab_art["vocabulary_sha256"]
NUM = json.load(open(HERE + "../g1/feature_schema_v2.json"))["raw_numerical_feature_names"]
SUBSET_IDS = json.load(open(HERE + "../g2/g2_validation_subset_row_ids.json"))


def load_subset(N):
    s, sha = load_train_sample()
    sub = s[s["id"].isin(SUBSET_IDS[str(N)]["row_ids"])].sort_values("id").reset_index(drop=True)
    pre = pp.PreprocessorV2(NUM, VOCAB).fit_scaler(sub)  # scaler fitted on the subset only
    X = pre.transform(sub)
    t = np.where(sub["label"].to_numpy() == 1, 1, -1)
    K = ek.sigmoid_gram(X, X)
    return dict(sub=sub, X=X, t=t, K=K, pre=pre, sample_sha=sha)


def refined_solve(M, B, steps=2):
    """Solve M Y = B by LU plus iterative refinement with long-double residuals."""
    Y = np.linalg.solve(M, B)
    Ml, Bl = M.astype(np.longdouble), B.astype(np.longdouble)
    for _ in range(steps):
        R = (Bl - Ml @ Y.astype(np.longdouble)).astype(np.float64)
        Y = Y + np.linalg.solve(M, R)
    return Y


def build_problem(K, t, ell, d=D):
    """Objects of twin problem `ell` with the accurately computed, symmetrized dense H."""
    N = K.shape[0]
    zeta = 1.0 if -ell == 1 else -1.0
    I_ell, I_opp = np.where(t == ell)[0], np.where(t == -ell)[0]
    S_ell = np.vstack([K[:, I_ell], np.ones((1, len(I_ell)))])
    S_opp = np.vstack([K[:, I_opp], np.ones((1, len(I_opp)))])
    M = S_ell @ S_ell.T + d * np.eye(N + 1)
    Y_lu = np.linalg.solve(M, S_opp)
    Y = refined_solve(M, S_opp)
    H_raw = S_opp.T @ Y
    H = 0.5 * (H_raw + H_raw.T)
    H_lu = 0.5 * (S_opp.T @ Y_lu + (S_opp.T @ Y_lu).T)
    info = {"H_raw_asymmetry_relative": float(np.linalg.norm(H_raw - H_raw.T) / np.linalg.norm(H_raw)),
            "max_abs_H_refined_minus_H_lu": float(np.max(np.abs(H - H_lu))), "max_abs_H": float(np.max(np.abs(H))), "n": int(H.shape[0])}
    return dict(N=N, zeta=zeta, I_ell=I_ell, I_opp=I_opp, S_ell=S_ell, S_opp=S_opp, M=M, Y=Y, H=H, H_lu=H_lu, info=info)


def natural_residual(H, alpha, c=C):
    g = H @ alpha - 1.0
    return float(np.max(np.abs(alpha - np.clip(alpha - g, 0.0, c))))


def certificates(P, alpha, c=C, d=D, project=False):
    """Independent validator. `P` from build_problem. Returns diagnostics for the given alpha.

    If project=True, alpha is first projected onto [0, c] and the violation of the raw alpha is reported.
    """
    H, S_ell, S_opp, M, Y, zeta = P["H"], P["S_ell"], P["S_opp"], P["M"], P["Y"], P["zeta"]
    out = {}
    a_raw = np.asarray(alpha, dtype=np.float64)
    out["bound_violation_raw"] = float(np.max(np.maximum(0.0, np.maximum(-a_raw, a_raw - c))))
    a = np.clip(a_raw, 0.0, c) if project else a_raw
    out["evaluated_on"] = "alpha projected onto [0,c]" if project else "alpha as given"
    out["projection_max_change"] = float(np.max(np.abs(a - a_raw)))
    g = H @ a - 1.0
    out["natural_residual_inf"] = float(np.max(np.abs(a - np.clip(a - g, 0.0, c))))
    D_val = float(a.sum() - 0.5 * a @ (H @ a))
    out["dual_objective_D"] = D_val
    rho = zeta * (Y @ a)                       # refined M^{-1} S_opp a
    out["stationarity_residual_inf"] = float(np.max(np.abs(M @ rho - zeta * (S_opp @ a))))
    z = zeta * (S_opp.T @ rho)                 # constraint left-hand side
    xi = np.maximum(0.0, 1.0 - z)
    out["primal_min_slack_max"] = float(xi.max())
    Pval = 0.5 * d * float(rho @ rho) + 0.5 * float(np.sum((S_ell.T @ rho) ** 2)) + c * float(xi.sum())
    out["primal_objective_P"] = Pval
    gap = Pval - D_val
    out["duality_gap"] = float(gap)
    out["duality_gap_relative"] = float(gap / max(1.0, abs(D_val)))
    # closed form derived in the audit: gap = alpha^T g + c * sum(max(0, -g_i))
    out["duality_gap_closed_form"] = float(a @ g + c * np.sum(np.maximum(0.0, -g)))
    out["closed_form_minus_numerical_gap"] = float(out["duality_gap_closed_form"] - gap)
    out["complementarity_alpha_inf"] = float(np.max(np.abs(a * (z + xi - 1.0))))
    out["complementarity_slack_inf"] = float(np.max(np.abs((c - a) * xi)))
    out["certified_rho_error_bound"] = float(np.sqrt(max(gap, 0.0) * 2.0 / d))
    out["rho"] = rho
    out["alpha_evaluated"] = a
    return out


def passes(cert):
    """Acceptance: natural residual and relative duality gap (thresholds unchanged). The bound violation of the
    raw alpha is required only when alpha is evaluated as given (not when a projection was applied and reported)."""
    ok = cert["natural_residual_inf"] <= TH["natural_residual_inf"] and cert["duality_gap_relative"] <= TH["relative_duality_gap"]
    if not cert["evaluated_on"].startswith("alpha projected"):
        ok = ok and cert["bound_violation_raw"] <= TH["bound_violation"]
    return bool(ok)


def strip(cert):
    return {k: v for k, v in cert.items() if k not in ("rho", "alpha_evaluated")}
