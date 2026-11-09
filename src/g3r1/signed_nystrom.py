"""Signed (indefinite) Nystrom factorization, as specified in g3_nystrom_specification.md.
Established technique (Oglic & Gartner 2019; Cai et al. 2021; Park et al. 2022); not a contribution of this project.
K_hat = Phi J Phi^T = C U_R Lambda_R^{-1} U_R^T C^T. No |K|, no clipping, no spectral majorant."""
import numpy as np

TAU_REL = 1e-10


def signed_nystrom(C, L, tau_rel=TAU_REL):
    """C = K[:, L] (N x m); L = ordered landmark row positions (length m)."""
    L = np.asarray(L)
    W = C[L, :]                                              # m x m
    lam, U = np.linalg.eigh(W)                               # ascending eigenvalues
    tau = tau_rel * float(np.max(np.abs(lam)))
    R = np.abs(lam) > tau
    lamR, UR = lam[R], U[:, R]
    scale = UR / np.sqrt(np.abs(lamR))                       # U_R |Lambda_R|^{-1/2}  (m x r)
    Phi = C @ scale                                          # N x r
    J = np.sign(lamR)                                        # +/-1 on retained components
    return {"L": L, "m": int(len(L)), "eigenvalues_W": lam, "tau_W": tau, "retained": R, "r": int(R.sum()), "n_plus": int((lam > tau).sum()), "n_minus": int((lam < -tau).sum()),
            "n_zero": int((~R).sum()), "Phi": Phi, "J": J, "scale": scale, "W_asymmetry_relative": float(np.linalg.norm(W - W.T) / max(np.linalg.norm(W), 1e-300))}


def phi_new(C_new, ny):
    """phi(x) = |Lambda_R|^{-1/2} U_R^T c(x) for rows c(x) of C_new (n_new x m); returns n_new x r."""
    return C_new @ ny["scale"]


def khat_dense(ny):
    """K_hat = Phi J Phi^T (symmetrized numerically; small N only)."""
    Kh = (ny["Phi"] * ny["J"]) @ ny["Phi"].T
    return 0.5 * (Kh + Kh.T)
