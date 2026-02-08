from typing import Any, Dict, Optional

import numpy as np
import scipy.linalg as la

from src.utils import blockdiag


def em_refine(
    X: np.array,
    Psi: np.array,
    A_init: np.array,
    C_init: np.array,
    W_init: np.array,
    Q_init: Optional[np.array] = None,
    Rx_init: Optional[np.array] = None,
    Rpsi_init: Optional[np.array] = None,
    mu_init: Optional[np.array] = None,
    P_init: Optional[np.array] = None,
    n_iters: int = 3,
    tol: float = 1e-4,
    reg_A: float = 0.01,
    reg_C: float = 0.01,
    jitter: float = 1e-10,
    verbose: bool = False,
) -> Dict[str, Any]:
    d = X.shape[1]
    m, T = Psi.shape
    if T != X.shape[0]:
        raise ValueError('X and Psi must share the same T')

    A = np.asarray(A_init, dtype=float)
    C = np.asarray(C_init, dtype=float)
    W = np.asarray(W_init, dtype=float)
    r = A.shape[0]

    if Q_init is None:
        Q = 1e-3 * np.eye(r)
    else:
        Q = np.asarray(Q_init, dtype=float)

    if Rx_init is None:
        Rx = 1e-2 * np.eye(d)
    else:
        Rx = np.asarray(Rx_init, dtype=float)

    if Rpsi_init is None:
        Rpsi = 1e-2 * np.eye(m)
    else:
        Rpsi = np.asarray(Rpsi_init, dtype=float)

    if mu_init is None:
        mu = np.zeros((r,), dtype=float)
    else:
        mu = np.asarray(mu_init, dtype=float).ravel()

    if P_init is None:
        P = np.eye(r, dtype=float)
    else:
        P = np.asarray(P_init, dtype=float)

    Y = np.vstack([X.T, Psi])  # (d+m, T)

    loglik_hist = []
    last_loglik = -np.inf

    for it in range(int(n_iters)):
        H = np.vstack([C, W])  # (d+m, r)
        R = blockdiag(Rx, Rpsi)  # (d+m, d+m)

        filt = kalman_filter(Y, A, H, Q, R, mu, P, jitter=jitter)
        loglik = filt['loglik']
        loglik_hist.append(loglik)

        if verbose:
            print(f'[EM] iter={it:02d} loglik={loglik:.6f}')

        if it > 0:
            denom = max(1.0, abs(last_loglik))
            if (loglik - last_loglik) / denom < tol:
                last_loglik = loglik
                break
        last_loglik = loglik

        sm = rts_smoother(A, filt, jitter=jitter)
        mu_s = sm['mu_smooth']  # (T,r)
        P_s = sm['P_smooth']  # (T,r,r)
        J = sm['J']  # (T-1,r,r)

        # Precompute E[z_t z_t^T]
        Ezz = np.zeros((T, r, r), dtype=float)
        for t in range(T):
            mt = mu_s[t].reshape(r, 1)
            Ezz[t] = P_s[t] + mt @ mt.T

        # Sums for A, H
        S_zz_0 = np.sum(Ezz[:-1], axis=0)
        S_zz_all = np.sum(Ezz, axis=0)
        S_zpzt = np.zeros((r, r), dtype=float)
        for t in range(T - 1):
            mt1 = mu_s[t + 1].reshape(r, 1)
            mt0 = mu_s[t].reshape(r, 1)
            S_zpzt += P_s[t + 1] @ J[t].T + mt1 @ mt0.T

        # Updates
        S_zz_0_reg = S_zz_0 + (T - 1) * reg_A * np.eye(r)
        A_new = S_zpzt @ la.inv(S_zz_0_reg)

        eigvals = np.linalg.eigvals(A_new)
        rho = np.max(np.abs(eigvals))
        if rho > 1.0001:
            A_new = A_new * (1.0001 / rho)

        S_zz_all_reg = S_zz_all + T * reg_C * np.eye(r)
        # sum y_t mu_t^T
        YMuT = Y @ mu_s  # (d+m, r)
        H_new = YMuT @ la.inv(S_zz_all_reg)

        # Q update
        S_zz_next = np.sum(Ezz[1:], axis=0)
        Q_new = (S_zz_next - A_new @ S_zpzt.T - S_zpzt @ A_new.T + A_new @ S_zz_0 @ A_new.T) / (T - 1)
        Q_new = 0.5 * (Q_new + Q_new.T)
        Q_new.flat[:: r + 1] += jitter

        # Split H into C and W
        C_new = H_new[:d, :]
        W_new = H_new[d:, :]

        # R_x and R_psi updates
        Rx_new = np.zeros((d, d), dtype=float)
        Rpsi_new = np.zeros((m, m), dtype=float)
        for t in range(T):
            mt = mu_s[t]
            Pt = P_s[t]
            x_t = X[t]
            psi_t = Psi[:, t]
            res_x = (x_t - C_new @ mt).reshape(d, 1)
            res_psi = (psi_t - W_new @ mt).reshape(m, 1)

            Rx_new += (res_x @ res_x.T) + (C_new @ Pt @ C_new.T)
            Rpsi_new += (res_psi @ res_psi.T) + (W_new @ Pt @ W_new.T)

        Rx_new /= T
        Rpsi_new /= T
        Rx_new = 0.5 * (Rx_new + Rx_new.T)
        Rx_new += 1e-4 * np.eye(d)
        Rpsi_new = 0.5 * (Rpsi_new + Rpsi_new.T)
        Rx_new.flat[:: d + 1] += jitter
        Rpsi_new.flat[:: m + 1] += jitter

        # Initial state updates
        mu_new = mu_s[0].copy()
        P_new = P_s[0].copy()
        P_new = 0.5 * (P_new + P_new.T)
        P_new.flat[:: r + 1] += jitter

        A, C, W, Q, Rx, Rpsi, mu, P = A_new, C_new, W_new, Q_new, Rx_new, Rpsi_new, mu_new, P_new

    return {
        'A': A,
        'C': C,
        'W': W,
        'Q': Q,
        'Rx': Rx,
        'Rpsi': Rpsi,
        'mu0': mu,
        'P0': P,
        'loglik_hist': np.array(loglik_hist, dtype=float),
        'S1': S_zz_all / T,
        'S2': S_zpzt / (T - 1),
        'S3': (Y @ mu_s) / T,
        'mu_smooth': mu_s,
        'P_smooth': P_s,
    }


def kalman_filter(
    Y: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    mu0: np.ndarray,
    P0: np.ndarray,
    jitter: float = 1e-10,
) -> Dict[str, Any]:
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError('Y must be (p,T)')
    p, T = Y.shape
    A = np.asarray(A, dtype=float)
    H = np.asarray(H, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    mu0 = np.asarray(mu0, dtype=float).ravel()
    P0 = np.asarray(P0, dtype=float)

    r = A.shape[0]
    I_r = np.eye(r)

    mu_pred = np.zeros((T, r), dtype=float)
    mu_filt = np.zeros((T, r), dtype=float)
    P_pred = np.zeros((T, r, r), dtype=float)
    P_filt = np.zeros((T, r, r), dtype=float)
    K_all = np.zeros((T, r, p), dtype=float)

    loglik = 0.0

    mu_pred[0] = mu0
    P_pred[0] = P0

    for t in range(T):
        y_t = Y[:, t]

        HP = H @ P_pred[t]
        S = HP @ H.T + R
        S = 0.5 * (S + S.T)
        S.flat[:: S.shape[0] + 1] += jitter

        for i in range(10):
            try:
                current_S = S + (i * jitter) * np.eye(p)
                L = la.cholesky(current_S, lower=True, check_finite=False)
                PHt = P_pred[t] @ H.T  # (r,p)
                Kt = la.cho_solve((L, True), PHt.T).T
                K_all[t] = Kt
                break
            except la.LinAlgError:
                continue

        innov = y_t - H @ mu_pred[t]
        mu_filt[t] = mu_pred[t] + Kt @ innov

        KH = Kt @ H
        IKH = I_r - KH
        P_f = IKH @ P_pred[t] @ IKH.T + Kt @ R @ Kt.T
        P_f = 0.5 * (P_f + P_f.T)
        P_filt[t] = P_f

        alpha = la.solve_triangular(L, innov, lower=True, check_finite=False)
        quad = alpha @ alpha
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        loglik += -0.5 * (p * np.log(2.0 * np.pi) + logdet + quad)

        if t + 1 < T:
            mu_pred[t + 1] = A @ mu_filt[t]
            P_pred[t + 1] = A @ P_filt[t] @ A.T + Q
            P_pred[t + 1] = 0.5 * (P_pred[t + 1] + P_pred[t + 1].T)

    return {
        'mu_pred': mu_pred,
        'P_pred': P_pred,
        'mu_filt': mu_filt,
        'P_filt': P_filt,
        'K': K_all,
        'loglik': loglik,
    }


def rts_smoother(
    A: np.ndarray,
    filt: Dict[str, Any],
    jitter: float = 1e-10,
) -> Dict[str, Any]:
    A = np.asarray(A, dtype=float)
    mu_pred = np.asarray(filt['mu_pred'], dtype=float)
    P_pred = np.asarray(filt['P_pred'], dtype=float)
    mu_filt = np.asarray(filt['mu_filt'], dtype=float)
    P_filt = np.asarray(filt['P_filt'], dtype=float)

    T, r = mu_filt.shape
    mu_smooth = mu_filt.copy()
    P_smooth = P_filt.copy()
    J = np.zeros((T - 1, r, r), dtype=float)

    for t in range(T - 2, -1, -1):
        Pp = P_pred[t + 1]
        Pp = 0.5 * (Pp + Pp.T)
        Pp.flat[:: r + 1] += jitter
        try:
            J_t = P_filt[t] @ A.T @ la.inv(Pp)
        except la.LinAlgError:
            J_t = P_filt[t] @ A.T @ la.pinv(Pp)
        J[t] = J_t

        mu_smooth[t] = mu_filt[t] + J_t @ (mu_smooth[t + 1] - mu_pred[t + 1])
        P_smooth[t] = P_filt[t] + J_t @ (P_smooth[t + 1] - P_pred[t + 1]) @ J_t.T
        P_smooth[t] = 0.5 * (P_smooth[t] + P_smooth[t].T)

    return {'mu_smooth': mu_smooth, 'P_smooth': P_smooth, 'J': J}
