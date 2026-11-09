"""Matrix-free Nystrom-KTSVM operator with the SYMMETRIC-SPD reduced formulation (g3r1_symmetric_Q_derivation.md).
Q = L (dI + L^T G L)^{-1} L^T with B = Phi^T Phi (+bias 1) = L L^T. The production path never forms Q, never inverts explicitly,
never symmetrizes as a fix, and adds no jitter/ridge. Cholesky failures raise CholeskyFailure (the run must stop)."""
import numpy as np
from scipy.linalg import cholesky, cho_factor, cho_solve, LinAlgError


class CholeskyFailure(Exception):
    pass


class TwinOperatorSPD:
    def __init__(self, Phi, J, t, ell, d=1e-3):
        N, r = Phi.shape
        self.N, self.r, self.d, self.ell = N, r, d, ell
        self.zeta = 1.0 if -ell == 1 else -1.0
        self.I_ell, self.I_opp = np.where(t == ell)[0], np.where(t == -ell)[0]
        self.Z_l = np.vstack([J[:, None] * Phi[self.I_ell].T, np.ones((1, len(self.I_ell)))])     # (r+1) x N_l
        self.Z_o = np.vstack([J[:, None] * Phi[self.I_opp].T, np.ones((1, len(self.I_opp)))])     # (r+1) x N_l'
        B = np.zeros((r + 1, r + 1))
        B[:r, :r] = Phi.T @ Phi
        B[r, r] = 1.0
        try:
            self.L = cholesky(B, lower=True)                                                     # B = L L^T
        except LinAlgError as e:
            raise CholeskyFailure("B CHOLESKY FAILURE: " + str(e))
        Y = self.L.T @ self.Z_l                                                                   # (r+1) x N_l
        A = d * np.eye(r + 1) + Y @ Y.T                                                           # = dI + L^T G L
        try:
            self.cfA = cho_factor(A, lower=True)                                                  # A = R R^T
        except LinAlgError as e:
            raise CholeskyFailure("A CHOLESKY FAILURE: " + str(e))
        self._B, self._A = B, A                                                                   # kept for diagnostics only
        self.n = len(self.I_opp)

    def _reduced_apply(self, a):
        b = self.L.T @ a                       # step 2
        y = cho_solve(self.cfA, b)             # step 3 (triangular solves)
        return self.L @ y                      # step 4

    def hvp(self, v):
        a = self.Z_o @ v                       # step 1
        return self.Z_o.T @ self._reduced_apply(a)   # step 5

    def reconstruct(self, alpha):
        return self.zeta * self._reduced_apply(self.Z_o @ alpha)

    @staticmethod
    def outputs_train(Phi, J, ub):
        return Phi @ (J * ub[:-1]) + ub[-1]

    @staticmethod
    def outputs_new(phi_new, J, ub):
        return phi_new @ (J * ub[:-1]) + ub[-1]

    def nbytes(self):
        return {"Z_l": int(self.Z_l.nbytes), "Z_opp": int(self.Z_o.nbytes), "L": int(self.L.nbytes), "chol_A": int(self.cfA[0].nbytes)}

    def diagnostics(self):
        """Validation only: assemble Q_spd = L A^{-1} L^T and report symmetry / conditioning."""
        Qs = self.L @ cho_solve(self.cfA, self.L.T)
        return {"Q_asymmetry_relative": float(np.linalg.norm(Qs - Qs.T) / max(np.linalg.norm(Qs), 1e-300)), "cond_B": float(np.linalg.cond(self._B)), "cond_A": float(np.linalg.cond(self._A)),
                "lambda_min_A": float(np.linalg.eigvalsh(self._A)[0]), "lambda_min_A_ge_d": bool(np.linalg.eigvalsh(self._A)[0] >= self.d * (1 - 1e-12)), "Q_finite": bool(np.isfinite(Qs).all()), "Q_shape": list(Qs.shape)}
