"""Exact KTSVM (published formulation) for the UNSW-NB15 scalability study.

Implements g2_ktsvm_specification.md (written before this file). The equation-to-code map is in
Section 3 of that specification. Deliberately NOT included: any decision rule, |K|, spectral-majorant
normalization, split regularization (d_w = d_b = d), K -> lambda K, or Nystrom.
"""
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

GAMMA = 10.0 / 148.0  # inherited fixed numerical hyperparameter (NOT recomputed from the new dimension)
THETA = -1.0          # inherited fixed configuration


def preactivation(X, Z, gamma=GAMMA, theta=THETA):
    return gamma * (X @ Z.T) + theta


def sigmoid_gram(X, Z, gamma=GAMMA, theta=THETA):
    """kappa(x, z) = tanh(gamma x^T z + theta), evaluated for all pairs."""
    return np.tanh(preactivation(X, Z, gamma, theta))


def _stack_S(K, idx):
    """S = [K[:, idx] ; 1^T]  with shape (N+1, len(idx))."""
    return np.vstack([K[:, idx], np.ones((1, len(idx)))])


def build_twin(K, t, ell, d):
    """Return dictionary with the objects of twin problem `ell` (spec Sections 1-2)."""
    N = K.shape[0]
    I_ell = np.where(t == ell)[0]
    I_opp = np.where(t == -ell)[0]
    S_ell = _stack_S(K, I_ell)               # (N+1, N_ell)
    S_opp = _stack_S(K, I_opp)               # (N+1, N_opp)
    M = S_ell @ S_ell.T + d * np.eye(N + 1)  # (N+1, N+1), symmetric positive definite for d > 0
    cf = cho_factor(M, lower=True)           # raises LinAlgError if M is not positive definite
    Y = cho_solve(cf, S_opp)                 # M^{-1} S_opp, (N+1, N_opp)
    H_raw = S_opp.T @ Y                      # (N_opp, N_opp)
    H = 0.5 * (H_raw + H_raw.T)
    asym = float(np.linalg.norm(H_raw - H_raw.T) / max(np.linalg.norm(H_raw), 1e-300))
    return dict(I_ell=I_ell, I_opp=I_opp, M=M, Y=Y, H=H, H_asym_rel=asym, zeta_opp=(1.0 if -ell == 1 else -1.0))


class ExactKTSVM:
    def __init__(self, c=1.0, d=1e-3, maxiter=5000, ftol=1e-12, gtol=1e-8, maxls=50):
        self.c, self.d = float(c), float(d)
        self.options = dict(maxiter=maxiter, ftol=ftol, gtol=gtol, maxls=maxls)

    def fit(self, K, t):
        t = np.asarray(t)
        if not np.all(np.isin(t, (-1, 1))):
            raise ValueError("labels must be +1/-1")
        N = K.shape[0]
        self.N = N
        self.alpha, self.rho, self.w, self.b, self.info = {}, {}, {}, {}, {}
        for ell in (1, -1):
            t0 = time.perf_counter()
            tw = build_twin(K, t, ell, self.d)
            H = tw["H"]
            n = H.shape[0]

            def fun(a, H=H):
                Ha = H @ a
                return 0.5 * float(a @ Ha) - float(a.sum()), Ha - 1.0

            t1 = time.perf_counter()
            res = minimize(fun, np.zeros(n), jac=True, method="L-BFGS-B", bounds=[(0.0, self.c)] * n, options=self.options)
            t2 = time.perf_counter()
            a = res.x
            rho = tw["zeta_opp"] * (tw["Y"] @ a)
            self.alpha[ell], self.rho[ell] = a, rho
            self.w[ell], self.b[ell] = rho[:N], float(rho[N])
            self.info[ell] = dict(
                n_dual_variables=int(n), n_class=int(len(tw["I_ell"])), n_opposite=int(len(tw["I_opp"])),
                scipy_success=bool(res.success), scipy_status=int(res.status), scipy_message=str(res.message),
                iterations=int(res.nit), function_evaluations=int(res.nfev), scipy_dual_objective_q=float(res.fun),
                H_asymmetry_relative=tw["H_asym_rel"], seconds_build=t1 - t0, seconds_solve=t2 - t1,
            )
        return self

    def raw_outputs(self, Kx):
        """Raw hyperplane outputs f_+(x), f_-(x) for rows of Kx = kappa(x_new, X_train), shape (n_new, N)."""
        return {ell: Kx @ self.w[ell] + self.b[ell] for ell in (1, -1)}


def verify_twin(K, t, ell, alpha, rho, c, d, eig_checks=True):
    """Independent certificates for twin problem `ell` (spec Section 5).

    Uses LU solves (np.linalg.solve) and recomputes everything from K, t, alpha, rho, so it does not
    share the Cholesky path of ExactKTSVM.fit. Returns a dictionary of diagnostics.
    """
    t = np.asarray(t)
    N = K.shape[0]
    zeta = 1.0 if -ell == 1 else -1.0
    I_ell, I_opp = np.where(t == ell)[0], np.where(t == -ell)[0]
    S_ell, S_opp = _stack_S(K, I_ell), _stack_S(K, I_opp)
    M = S_ell @ S_ell.T + d * np.eye(N + 1)
    Y = np.linalg.solve(M, S_opp)
    H = S_opp.T @ Y
    Hs = 0.5 * (H + H.T)
    g = Hs @ alpha - 1.0
    nat = alpha - np.clip(alpha - g, 0.0, c)
    out = {}
    out["dual_dimension"] = int(len(alpha))
    out["natural_residual_inf"] = float(np.max(np.abs(nat)))
    out["bound_violation"] = float(np.max(np.maximum(0.0, np.maximum(-alpha, alpha - c))))
    out["n_at_lower_bound"] = int(np.sum(alpha <= 0.0))
    out["n_at_upper_bound"] = int(np.sum(alpha >= c))
    out["n_strictly_inside"] = int(np.sum((alpha > 0.0) & (alpha < c)))
    D = float(alpha.sum() - 0.5 * alpha @ (Hs @ alpha))
    out["dual_objective_D"] = D
    rhs = zeta * (S_opp @ alpha)
    out["stationarity_residual_inf_absolute"] = float(np.max(np.abs(M @ rho - rhs)))
    out["stationarity_residual_inf_relative"] = out["stationarity_residual_inf_absolute"] / max(1.0, float(np.max(np.abs(rhs))))
    rho_lu = zeta * (Y @ alpha)
    out["rho_difference_between_solver_path_and_LU_path_inf"] = float(np.max(np.abs(rho - rho_lu)))
    z = zeta * (S_opp.T @ rho)
    xi = np.maximum(0.0, 1.0 - z)
    out["primal_min_slack_max"] = float(xi.max())
    P = 0.5 * d * float(rho @ rho) + 0.5 * float(np.sum((S_ell.T @ rho) ** 2)) + c * float(xi.sum())
    out["primal_objective_P"] = P
    gap = P - D
    out["duality_gap"] = float(gap)
    out["duality_gap_relative"] = float(gap / max(1.0, abs(D)))
    out["complementarity_alpha_inf"] = float(np.max(np.abs(alpha * (z + xi - 1.0))))
    out["complementarity_slack_inf"] = float(np.max(np.abs((c - alpha) * xi)))
    out["certified_rho_error_bound_sqrt_2gap_over_d"] = float(np.sqrt(max(gap, 0.0) * 2.0 / d))
    out["rho_norm_2"] = float(np.linalg.norm(rho))
    if eig_checks:
        lh = np.linalg.eigvalsh(Hs)
        lm = np.linalg.eigvalsh(M)
        out["H_lambda_min"], out["H_lambda_max"] = float(lh[0]), float(lh[-1])
        out["H_psd_relative_min"] = float(lh[0] / max(lh[-1], 1e-300))
        out["M_lambda_min"], out["M_lambda_max"] = float(lm[0]), float(lm[-1])
        out["M_condition_number_2"] = float(lm[-1] / lm[0])
    out["H_asymmetry_relative_LU_path"] = float(np.linalg.norm(H - H.T) / max(np.linalg.norm(H), 1e-300))
    return out
