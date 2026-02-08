from typing import Any, Dict, Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2

from src.models.adakoop.module.kernel import build_kernel
from src.models.adakoop.module.lks import LKS
from src.models.adakoop.module.storage import Storage

MAX_MODELS = 30


class AdaKoop:
    def __init__(self, verbose: bool = True, max_models: int = MAX_MODELS) -> None:
        self.verbose = verbose
        self.max_models = max_models

    def init_params(
        self,
        d: int,
        lcurr: int,
        nu: float,
        gamma: float,
        kernel_type: str,
        lambda_A: float = 1e-6,
        em_iters: int = 3,
        em_tol: float = 1e-6,
        r_init: float = 1e-2,
        m_max: int = 100,
        burnin: int = 30,
        chi2_p: float = 0.99,
        cusum_h: Optional[float] = None,
        exceed_rate_th: float = 0.05,
        state_reset_P_scale: float = 1e3,
        jitter: float = 1e-8,
        compress: bool = True,
        add_dict: bool = True,
        online_update: bool = True,
    ) -> None:
        self.d = d
        self.lcurr = lcurr
        self.nu = nu
        self.gamma = gamma
        self.kernel_type = kernel_type
        self.lambda_A = lambda_A
        self.em_iters = em_iters
        self.em_tol = em_tol
        self.r_init = r_init
        self.burnin = burnin
        self.m_max = m_max
        self.jitter = jitter

        self.chi2_p = chi2_p
        self.chi2_th = chi2.ppf(self.chi2_p, df=self.d)
        self.cusum_h = 3 * self.chi2_th if cusum_h is None else cusum_h * self.chi2_th
        self.least_duration = lcurr // 2
        self.exceed_rate_th = exceed_rate_th
        self.state_reset_P_scale = state_reset_P_scale

        self.lks_c = None
        self._lks_c_idx = None
        self._create_cnt = 0
        self._cusum_g = 0.0
        self._skip_next_update = False

        self.compress = compress
        self.add_dict = add_dict
        self.online_update = online_update

    def initialize(
        self,
        X_init: np.ndarray,
        n_clusters: Optional[int] = None,
        dist_threshold: Optional[float] = None,
    ) -> None:
        T_init = X_init.shape[0]

        kernel = build_kernel(X_init, kernel_type=self.kernel_type)
        self.storage = Storage(
            nu=self.nu,
            gamma=self.gamma,
            lambda_A=self.lambda_A,
            em_iters=self.em_iters,
            em_tol=self.em_tol,
            r_init=self.r_init,
            m_max=self.m_max,
            compress=self.compress,
            jitter=self.jitter,
            verbose=self.verbose,
        )

        windows, indices = [], []
        for t in range(0, T_init - self.lcurr + 1, self.lcurr // 2):
            windows.append(X_init[t : t + self.lcurr])
            indices.append(t)

        last_start = T_init - self.lcurr
        if len(indices) == 0 or indices[-1] != last_start:
            windows.append(X_init[last_start : last_start + self.lcurr])
            indices.append(last_start)

        n_wins = len(windows)
        k2_self = np.empty((n_wins,), dtype=float)
        for i in range(n_wins):
            k2 = self._transition_kernel(windows[i], windows[i], kernel=kernel)
            k2_self[i] = max(k2, 0.0)

        dist_list = []
        for i in range(n_wins):
            for j in range(i + 1, n_wins):
                k2_ij = self._transition_kernel(windows[i], windows[j], kernel=kernel)
                denom = (self.jitter + k2_self[i]) * (self.jitter + k2_self[j])
                a2 = ((self.jitter + k2_ij) ** 2) / denom
                a2 = np.clip(a2, 0.0, 1.0)
                d_val = np.sqrt(max(0.0, 1.0 - a2))
                dist_list.append(d_val)
        dist_vec = np.asarray(dist_list, dtype=float)

        Z = linkage(dist_vec, method='average')
        if n_clusters is not None:
            n_clusters = min(n_clusters, self.max_models)
            labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        else:
            if dist_threshold is None:
                dist_threshold = np.median(dist_vec) * 1.5
            labels = fcluster(Z, t=dist_threshold, criterion='distance')

            if len(np.unique(labels)) > self.max_models:
                labels = fcluster(Z, t=self.max_models, criterion='maxclust')

        unique_labels = np.unique(labels)
        if self.verbose:
            print(f'Created {len(unique_labels)} initial models from {n_wins} windows.')

        runs_by_label = {}
        i = 0
        while i < n_wins:
            lbl_i = labels[i]
            j = i
            while (j + 1) < n_wins and labels[j + 1] == lbl_i:
                j += 1
            runs_by_label.setdefault(lbl_i, []).append((i, j))
            i = j + 1

        last_window_idx = n_wins - 1
        last_label = labels[last_window_idx]

        def _segment_from_run(win_start_idx: int, win_end_idx: int) -> np.ndarray:
            start_t = indices[win_start_idx]
            end_t = indices[win_end_idx] + self.lcurr
            if end_t <= start_t:
                start_t = max(0, min(start_t, T_init - self.lcurr))
                end_t = min(T_init, start_t + self.lcurr)
            return X_init[start_t:end_t]

        tail_start_idx = last_window_idx
        while tail_start_idx > 0 and labels[tail_start_idx - 1] == last_label:
            tail_start_idx -= 1
        tail_Xc = _segment_from_run(tail_start_idx, last_window_idx)

        sq_dist_mat = squareform(dist_vec)
        for lbl in unique_labels:
            lbl_int = lbl
            members = np.where(labels == lbl_int)[0]
            if members.size == 0:
                continue

            sub_mat = sq_dist_mat[np.ix_(members, members)]
            sum_dists = np.sum(sub_mat, axis=1)
            medoid_local = np.argmin(sum_dists)
            medoid_global = members[medoid_local]

            Xc_train = windows[medoid_global]
            train_note = f'medoid_window idx={medoid_global}'

            if lbl_int == last_label:
                Xc_train = tail_Xc
                train_note = f'tail_run T={Xc_train.shape[0]}'
            else:
                runs = runs_by_label.get(lbl_int, [])
                if len(runs) > 0:
                    best_run = max(runs, key=lambda ab: (ab[1] - ab[0] + 1, ab[1]))
                    run_len = best_run[1] - best_run[0] + 1
                    if run_len >= 2:
                        Xc_train = _segment_from_run(best_run[0], best_run[1])
                        train_note = f'longest_run win=[{best_run[0]},{best_run[1]}] T={Xc_train.shape[0]}'

            self.storage.create(
                Xc=Xc_train,
                kernel=kernel,
                append=True,
            )

            if self.verbose:
                print(f'label={lbl_int} trained on {train_note}')

    def model_selection(self, x_new: np.ndarray) -> Dict[str, Any]:
        self._create_cnt += 1
        created_new = False
        self.current_data = np.vstack([self.current_data, x_new])
        if len(self.current_data) > self.lcurr:
            self.current_data = self.current_data[-self.lcurr :]

        if self._in_recovery:
            if len(self.current_data) < self.lcurr:
                return {
                    'mode': 'recovery',
                    'active_model_idx': self.lks_c.idx,
                    'status': 'accumulating_data',
                    'current_len': len(self.current_data),
                }
            else:
                best_score = float('inf')
                accepted_any = False
                for lks in self.storage():
                    metrics = lks.score_window(
                        Xc=self.current_data,
                        chi2_th=self.chi2_th,
                        reset_P_scale=self.state_reset_P_scale,
                        burnin=self.burnin,
                    )

                    exceed_rate = metrics['exceed_rate']
                    cusum_max = metrics['cusum_max']
                    mean_dist_sq = metrics['mean_dist_sq']

                    ok = (exceed_rate <= self.exceed_rate_th) and (cusum_max <= self.cusum_h)
                    if ok:
                        accepted_any = True

                    if mean_dist_sq < best_score:
                        best_score = mean_dist_sq

                if not accepted_any and (len(self.storage) < self.max_models):
                    if self._create_cnt < self.least_duration:
                        self.storage.pop()
                    kernel = build_kernel(self.current_data, kernel_type=self.kernel_type)
                    lks_new, _ = self.storage.create(
                        Xc=self.current_data,
                        kernel=kernel,
                        append=True,
                    )
                    self.lks_c = lks_new
                    self._lks_c_idx = lks_new.idx
                    self._create_cnt = 0
                    created_new = True

                    self._skip_next_update = True
                    self._cusum_g = 0.0
                    return {
                        'triggered': True,
                        'switched': created_new,
                        'created_new': created_new,
                        'cusum_g': self._cusum_g,
                        'active_model_idx': self.lks_c.idx,
                    }
                self._in_recovery = False

        dist_sq = self._innovation_dist_sq(self.lks_c, x_new)
        self._cusum_g = max(0.0, self._cusum_g + (dist_sq - self.chi2_th))
        triggered = bool(self._cusum_g > self.cusum_h)

        if not triggered:
            return {
                'triggered': False,
                'switched': created_new,
                'created_new': created_new,
                'cusum_g': self._cusum_g,
                'active_model_idx': self.lks_c.idx,
            }

        change_point = self.lks_c.detect_change_point(
            self.current_data,
            chi2_th=self.chi2_th,
            reset_P_scale=self.state_reset_P_scale,
        )
        self.current_data = self.current_data[change_point:]
        self._in_recovery = True
        self._cusum_g = 0.0

        return {
            'triggered': True,
            'created_new': created_new,
            'dist_sq': dist_sq,
            'cusum_g': self._cusum_g,
            'active_model_idx': self.lks_c.idx,
        }

    def update(self, x_new: np.ndarray) -> Dict[str, Any]:
        if self._skip_next_update:
            self._skip_next_update = False
            return {
                'skipped': True,
                'active_model_idx': self.lks_c.idx,
            }

        out = self.lks_c.update(
            x_new,
            compress=self.compress,
            update_dict=self.add_dict,
            update_params=self.online_update,
            update_state=True,
        )
        out['skipped'] = False
        out['active_model_idx'] = self.lks_c.idx
        return out

    def forecast(self, lstep: int, scale: float = 3.0) -> tuple[np.ndarray, np.ndarray, bool]:
        pred_mean, pred_cov = self.lks_c.predict(lstep)

        traces = np.trace(pred_cov, axis1=1, axis2=2)
        uncertainty = np.max(traces)
        data_var = np.sum(np.var(self.current_data, axis=0, ddof=1))
        var_threshold = data_var * scale
        if uncertainty > var_threshold:
            if self.verbose:
                print(f'Uncertainty {uncertainty:.4f} exceeded threshold {var_threshold:.4f}')
            current_x_est = self.lks_c.params['C'] @ self.lks_c.current_mu
            fallback_mean = np.tile(current_x_est, (lstep, 1))

            return fallback_mean, pred_cov, True

        return pred_mean, pred_cov, False

    def set_initial_model(self, X: np.ndarray) -> None:
        if self.lks_c is not None:
            print('[Warning] Initial model is already set. Overwriting.')

        best_score_any = float('inf')
        for pos, lks in enumerate(self.storage()):
            metrics = lks.score_window(
                Xc=X,
                chi2_th=self.chi2_th,
                reset_P_scale=self.state_reset_P_scale,
                burnin=self.burnin,
            )

            mean_dist_sq = metrics['mean_dist_sq']
            if mean_dist_sq < best_score_any:
                best_score_any = mean_dist_sq
                best_pos_any = pos
                best_metrics_any = metrics

        selected = self.storage[best_pos_any]
        _set_filter_state(selected, best_metrics_any['mu_before_last'], best_metrics_any['P_before_last'])
        self.lks_c = selected
        self._lks_c_idx = selected.idx
        self.current_data = X
        self._in_recovery = False

    def _transition_kernel(self, X: np.ndarray, Y: np.ndarray, kernel: Any) -> float:
        L = min(X.shape[0], Y.shape[0])
        X, Y = X[:L], Y[:L]

        K = kernel.pairwise(X, Y)  # (L, L)
        tr11 = np.trace(K[:-1, :-1])
        tr22 = np.trace(K[1:, 1:])
        term2 = np.sum(K[:-1, 1:] * K[1:, :-1].T)

        return tr11 * tr22 - term2

    def _innovation_dist_sq(self, lks: LKS, x_t: np.ndarray) -> float:
        out = lks.one_step_prediction(x_t)
        return out['dist_sq']


def _set_filter_state(lks: LKS, mu: np.ndarray, P: np.ndarray) -> None:
    mu = np.asarray(mu, dtype=float).ravel()
    P = np.asarray(P, dtype=float)
    lks.current_mu = mu.copy()
    lks.current_P = P.copy()
