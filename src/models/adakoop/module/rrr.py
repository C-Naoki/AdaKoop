from typing import Dict, Optional

import numpy as np
import scipy.linalg as la


def rrr(
    X: np.ndarray,
    Psi: np.ndarray,
    lambda_A: float = 1e-6,
    lambda_C: float = 1e-6,
    jitter: float = 1e-10,
) -> Dict[str, np.ndarray]:
    m, T = Psi.shape

    Psi0 = Psi[:, :-1]  # (m,T-1)
    Psi1 = Psi[:, 1:]  # (m,T-1)
    denom = T - 1

    S10 = (Psi1 @ Psi0.T) / denom  # (m,m)
    S00 = (Psi0 @ Psi0.T) / denom + lambda_A * np.eye(m)  # (m,m)

    # S00^{-1/2} via eigen decomposition (SPD)
    eigvals, eigvecs = la.eigh(S00)
    eigvals = np.maximum(eigvals, jitter)
    S00_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T  # (m,m)

    M = S10 @ S00_inv_sqrt  # (m,m)
    U_r, s_r, Vt_r, r_opt = svd(M)

    W_hat = U_r  # (m,r)
    A_hat = np.diag(s_r) @ Vt_r @ S00_inv_sqrt @ U_r  # (r,r)

    # Observation mapping: C = X * Z^T (Z Z^T + lambda I)^{-1}, Z = W^\dagger Psi
    W_pinv = la.pinv(W_hat)  # (r,m)
    Z = W_pinv @ Psi  # (r,T)
    ZZt = Z @ Z.T  # (r,r)
    C_hat = (X.T @ Z.T) @ la.inv(ZZt + T * lambda_C * np.eye(r_opt))  # (d,r)

    return {'A': A_hat, 'W': W_hat, 'C': C_hat}


def svd(M: np.ndarray, rank: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    U, s, Vt = la.svd(M, full_matrices=False)
    if rank is None:
        m, n = M.shape
        beta = min(m, n) / max(m, n)
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        median_val = np.median(s)
        threshold = omega * median_val
        r_opt = np.sum(s > threshold)
        r_opt = max(r_opt, 1)
        return U[:, :r_opt], s[:r_opt], Vt[:r_opt, :], r_opt
    elif rank <= 1:
        s_squared = s**2
        total_energy = np.sum(s_squared)
        cumulative_energy = np.cumsum(s_squared) / total_energy
        print(cumulative_energy)
        r_opt = np.searchsorted(cumulative_energy, rank) + 1
        return U[:, :r_opt], s[:r_opt], Vt[:r_opt, :], r_opt
    else:
        return U[:, :rank], s[:rank], Vt[:rank, :], rank
