from typing import Any, Dict, Optional

import numpy as np
import scipy.linalg as la

from src.models.adakoop.module.ald import ALDDictionary
from src.models.adakoop.module.refiner import blockdiag, em_refine, kalman_filter, rts_smoother
from src.models.adakoop.module.rrr import rrr


class DKS:
    def __init__(
        self,
        idx: int,
        ald: ALDDictionary,
        gamma: float = 0.02,
        lambda_A: float = 1e-6,
        lambda_C: float = 1e-6,
        em_iters: int = 3,
        em_tol: float = 1e-6,
        r_init: float = 1e-2,
        jitter: float = 1e-10,
        verbose: bool = True,
    ) -> None:
        self.idx = idx
        self.ald = ald
        self.dict = ald
        self.gamma = gamma
        self.lambda_A = lambda_A
        self.lambda_C = lambda_C
        self.em_iters = em_iters
        self.em_tol = em_tol
        self.jitter = jitter
        self.r_init = r_init
        self.verbose = verbose

        self._fitted = False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DKS):
            return self.idx == other.idx
        else:
            raise TypeError('Not supported type')

    def fit(self, data: np.ndarray, compress: bool = True) -> dict:
        T, self.d = data.shape

        # Step 1. Orthogonalization
        for t in range(T):
            self.ald.update(data[t], compress=compress)
        Psi = self.ald.compute_feature_matrix(data)  # (m,T)

        # Step 2. Coarse Initialization via RRR
        init = rrr(
            data,
            Psi,
            lambda_A=self.lambda_A,
            lambda_C=self.lambda_C,
            jitter=self.jitter,
        )
        A0, C0, W0 = init['A'], init['C'], init['W']

        # Step 3. EM Refinement
        params = em_refine(
            X=data,
            Psi=Psi,
            A_init=A0,
            C_init=C0,
            W_init=W0,
            n_iters=self.em_iters,
            tol=self.em_tol,
            reg_A=self.lambda_A,
            reg_C=self.lambda_C,
            jitter=self.jitter,
            verbose=self.verbose,
        )

        self.params = params
        self.r = self.params['A'].shape[0]

        self.current_mu = params['mu_smooth'][-1].copy()
        self.current_P = params['P_smooth'][-1].copy()

        self._fitted = True

        return params

    def predict(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError('Model is not fitted yet. Call fit() before predict().')

        A, C = self.params['A'], self.params['C']
        Q = self.params['Q']
        Rx = self.params['Rx']

        d = C.shape[0]

        z_curr = self.current_mu.copy()  # (r,)
        P_curr = self.current_P.copy()  # (r, r)

        X_pred = np.empty((n_steps, d), dtype=float)
        Cov_pred = np.empty((n_steps, d, d), dtype=float)

        for t in range(n_steps):
            z_next = A @ z_curr
            P_next = A @ P_curr @ A.T + Q
            P_next = 0.5 * (P_next + P_next.T)

            x_next = C @ z_next
            Sigma_next = C @ P_next @ C.T + Rx
            Sigma_next = 0.5 * (Sigma_next + Sigma_next.T)

            X_pred[t] = x_next
            Cov_pred[t] = Sigma_next

            z_curr = z_next
            P_curr = P_next

        return X_pred, Cov_pred

    def initial_update(
        self,
        Xc: np.ndarray,
        Psi: Optional[np.ndarray] = None,
        P_scale: float = 1e3,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError('Model is not fitted yet.')

        if Psi is None:
            Psi = self.ald.compute_feature_matrix(Xc)
        Y = np.vstack([Xc.T, Psi])  # (d+m, T_new)

        A = self.params['A']
        H = np.vstack([self.params['C'], self.params['W']])
        Q = self.params['Q']
        R = blockdiag(self.params['Rx'], self.params['Rpsi'])

        mu_dummy = np.zeros((self.r,), dtype=float)
        P_dummy = P_scale * np.eye(self.r, dtype=float)
        kf_out = kalman_filter(Y=Y, A=A, H=H, Q=Q, R=R, mu0=mu_dummy, P0=P_dummy, jitter=self.jitter)
        sm_out = rts_smoother(A=A, filt=kf_out, jitter=self.jitter)

        optimal_mu = sm_out['mu_smooth'][0].copy()
        optimal_P = sm_out['P_smooth'][0].copy()

        return optimal_mu, optimal_P

    def score_window(
        self,
        Xc: np.ndarray,
        chi2_th: float,
        reset_P_scale: float = 1e3,
        burnin: int = 30,
    ) -> Dict[str, Any]:
        res = self.run_filter_on_window(Xc, reset_P_scale)

        dist_arr = res['dist_seq']
        kf_out = res['kf_out']

        valid_dist = dist_arr[burnin:] if len(dist_arr) > burnin else dist_arr

        exceed_rate = np.mean(valid_dist > chi2_th)
        mean_dist = np.mean(valid_dist)

        g = 0.0
        g_max = 0.0
        for ds in dist_arr[burnin:]:
            g = max(0.0, g + (ds - chi2_th))
            g_max = max(g_max, g)

        mu_before_last = kf_out['mu_filt'][-1].copy()
        P_before_last = kf_out['P_filt'][-1].copy()

        return {
            'mean_dist_sq': mean_dist,
            'exceed_rate': exceed_rate,
            'cusum_max': g_max,
            'mu_before_last': mu_before_last.ravel(),
            'P_before_last': P_before_last,
            'dist_seq': dist_arr,
        }

    def detect_change_point(
        self,
        Xc: np.ndarray,
        chi2_th: float,
        reset_P_scale: float,
    ) -> int:
        res = self.run_filter_on_window(Xc, reset_P_scale)
        dist_seq = res['dist_seq']

        cusum_g = 0.0
        last_zero_idx = -1
        for t, dist_sq in enumerate(dist_seq):
            s_t = dist_sq - chi2_th
            if cusum_g + s_t <= 0:
                cusum_g = 0.0
                last_zero_idx = t
            else:
                cusum_g += s_t

        change_point = last_zero_idx + 1
        return min(change_point, len(Xc) - 1)

    def update(
        self,
        x_t: np.ndarray,
        compress: bool = True,
        update_dict: bool = True,
        update_params: bool = True,
        update_state: bool = True,
    ) -> Dict[str, Any]:
        m_prev = self.dict.size
        if update_dict:
            added, delta, a = self.dict.update(x_t, compress=compress)
        else:
            added, delta, a = False, 0.0, None

        if update_dict and added and (self.dict.size > m_prev):
            if a is None:
                new_row = np.zeros((1, self.params['W'].shape[1]), dtype=float)
                a_vec = np.zeros((0,), dtype=float)
            else:
                a_vec = np.asarray(a, dtype=float).ravel()
                new_row = (a_vec.reshape(1, -1) @ self.params['W']).reshape(1, -1)  # (1,r)

            self.params['W'] = np.vstack([self.params['W'], new_row])
            self.params['Rpsi'] = blockdiag(self.params['Rpsi'], np.array([[self.r_init]], dtype=float))

            S3_dict_part = self.params['S3'][self.d :, :]
            new_row_S3 = (a_vec.reshape(1, -1) @ S3_dict_part).reshape(1, -1)
            self.params['S3'] = np.vstack([self.params['S3'], new_row_S3])

        if update_dict and (self.dict.size > self.dict.m_max):
            removed_idx = self.dict.prune()
            if removed_idx != -1:
                self.params['W'] = np.delete(self.params['W'], removed_idx, axis=0)  # W
                Rpsi_temp = np.delete(self.params['Rpsi'], removed_idx, axis=0)  # Rpsi
                self.params['Rpsi'] = np.delete(Rpsi_temp, removed_idx, axis=1)
                self.params['S3'] = np.delete(self.params['S3'], self.d + removed_idx, axis=0)  # S3

        m = self.dict.size
        H = np.vstack([self.params['C'], self.params['W']])  # (d+m, r)
        R = blockdiag(self.params['Rx'], self.params['Rpsi'])  # (d+m, d+m)

        pred_res = self.one_step_prediction(x_t)
        current_mu_pred = pred_res['current_mu_pred']
        current_P_pred = pred_res['current_P_pred']
        x_pred = pred_res['x_pred']
        dist_sq = pred_res['dist_sq']
        psi_t = self.dict.feature(x_t)  # (m,)
        y_t = np.concatenate([x_t, psi_t], axis=0)  # (d+m,)
        y_pred = H @ current_mu_pred

        HP = H @ current_P_pred
        S = HP @ H.T + R
        S = 0.5 * (S + S.T)
        p = S.shape[0]
        S.flat[:: p + 1] += self.jitter

        try:
            L = la.cholesky(S, lower=True, check_finite=False)
        except la.LinAlgError:
            S = S + (10.0 * self.jitter) * np.eye(p)
            L = la.cholesky(S, lower=True, check_finite=False)

        PHt = current_P_pred @ H.T  # (r,p)
        Kt = la.cho_solve((L, True), PHt.T).T
        innov = y_t - y_pred

        mu_f = current_mu_pred + Kt @ innov
        KH = Kt @ H
        IKH = np.eye(self.r) - KH
        P_f = IKH @ current_P_pred @ IKH.T + Kt @ R @ Kt.T
        P_f = 0.5 * (P_f + P_f.T)

        x_filt = (self.params['C'] @ mu_f).copy()

        if update_params:
            S1_prev = self.params['S1'].copy()
            Ezz = P_f + np.outer(mu_f, mu_f)
            self.params['S1'] = (1.0 - self.gamma) * self.params['S1'] + self.gamma * Ezz

            P_cross = (np.eye(self.r) - Kt @ H) @ self.params['A'] @ self.current_P
            cross_approx = P_cross + np.outer(mu_f, self.current_mu)
            self.params['S2'] = (1.0 - self.gamma) * self.params['S2'] + self.gamma * cross_approx
            self.params['S3'] = (1.0 - self.gamma) * self.params['S3'] + self.gamma * np.outer(y_t, mu_f)
            S1_prev_reg = S1_prev + self.lambda_A * np.eye(self.r)
            self.params['A'] = self.params['S2'] @ la.inv(S1_prev_reg)

            eigvals = np.linalg.eigvals(self.params['A'])
            rho = np.max(np.abs(eigvals))
            if rho > 1.0001:
                self.params['A'] = self.params['A'] * (1.0001 / rho)

            S1_reg = self.params['S1'] + self.lambda_C * np.eye(self.r)
            H_new = self.params['S3'] @ la.pinv(S1_reg, rtol=1e-5)
            self.params['C'] = H_new[: self.d, :]
            self.params['W'] = H_new[self.d :, :]

        if update_state:
            self.current_mu = mu_f.copy()
            self.current_P = P_f.copy()

        return {
            'added': bool(added),
            'delta': delta,
            'm': m,
            'x_pred': x_pred,
            'x_filt': x_filt,
            'dist_sq': dist_sq,
        }

    def one_step_prediction(self, x_t: np.ndarray) -> Dict[str, Any]:
        mu_pred = self.params['A'] @ self.current_mu
        P_pred = self.params['A'] @ self.current_P @ self.params['A'].T + self.params['Q']
        P_pred = 0.5 * (P_pred + P_pred.T)

        C = self.params['C']
        x_pred = C @ mu_pred
        innov_x = x_t - x_pred

        Sx = C @ P_pred @ C.T + self.params['Rx']
        Sx = 0.5 * (Sx + Sx.T)

        dx = Sx.shape[0]
        Sx.flat[:: dx + 1] = np.maximum(Sx.flat[:: dx + 1], 1e-4)

        try:
            Lx = la.cholesky(Sx, lower=True, check_finite=False)
        except la.LinAlgError:
            Sx = Sx + (10.0 * self.jitter) * np.eye(dx)
            Lx = la.cholesky(Sx, lower=True, check_finite=False)

        dist_sq = innov_x @ la.cho_solve((Lx, True), innov_x)

        return {
            'dist_sq': dist_sq,
            'current_mu_pred': mu_pred,
            'current_P_pred': P_pred,
            'x_pred': x_pred,
        }

    def run_filter_on_window(
        self,
        Xc: np.ndarray,
        reset_P_scale: float = 1e3,
    ) -> Dict[str, Any]:
        Xc = np.asarray(Xc, dtype=float)
        T = Xc.shape[0]

        Psi = self.ald.compute_feature_matrix(Xc)
        Y = np.vstack([Xc.T, Psi])

        mu_init = np.zeros(self.r, dtype=float)
        P_init = reset_P_scale * np.eye(self.r, dtype=float)
        A = self.params['A']
        H = np.vstack([self.params['C'], self.params['W']])
        Q = self.params['Q']
        R = blockdiag(self.params['Rx'], self.params['Rpsi'])

        kf_out = kalman_filter(Y, A, H, Q, R, mu_init, P_init, jitter=self.jitter)

        mu_pred = kf_out['mu_pred']
        P_pred = kf_out['P_pred']

        dist_list = []
        C = self.params['C']
        Rx = self.params['Rx']

        for t in range(T):
            innov_x = Xc[t] - (C @ mu_pred[t])
            Sx = C @ P_pred[t] @ C.T + Rx
            Sx = 0.5 * (Sx + Sx.T)

            dx = Sx.shape[0]
            Sx.flat[:: dx + 1] += self.jitter

            try:
                Lx = la.cholesky(Sx, lower=True, check_finite=False)
                ds = innov_x @ la.cho_solve((Lx, True), innov_x)
            except la.LinAlgError:
                Sx_safe = Sx + (10.0 * self.jitter) * np.eye(dx)
                try:
                    Lx = la.cholesky(Sx_safe, lower=True)
                    ds = innov_x @ la.cho_solve((Lx, True), innov_x)
                except la.LinAlgError:
                    ds = np.float('inf')

            dist_list.append(ds)

        return {
            'dist_seq': np.array(dist_list, dtype=float),
            'kf_out': kf_out,
            'T': T,
        }
