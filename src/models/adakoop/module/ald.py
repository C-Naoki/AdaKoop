from typing import Any, Optional, Tuple

import numpy as np


class ALDDictionary:
    def __init__(
        self,
        kernel: Any,
        nu: float,
        m_max: int = 100,
        jitter: float = 1e-10,
    ) -> None:
        self.kernel = kernel
        self.nu = nu
        self.m_max = m_max
        self.jitter = jitter

        self._D: Optional[np.ndarray] = None  # (m,d)
        self._K_inv: Optional[np.ndarray] = None  # (m,m)

    @property
    def D(self) -> np.ndarray:
        if self._D is None:
            return np.zeros((0, 0), dtype=float)
        return self._D

    @property
    def K_inv(self) -> np.ndarray:
        if self._K_inv is None:
            return np.zeros((0, 0), dtype=float)
        return self._K_inv

    @property
    def size(self) -> int:
        return 0 if self._D is None else int(self._D.shape[0])

    def feature(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        if self.size == 0:
            return np.zeros((0,), dtype=float)
        Kdx = self.kernel.pairwise(self.D, x.reshape(1, -1))  # (m,1)
        return Kdx[:, 0].copy()

    def update(self, x: np.ndarray, compress: bool = True) -> Tuple[bool, float, Optional[np.ndarray]]:
        x = np.asarray(x, dtype=float).ravel()
        if self.size == 0:
            k_xx = self.kernel(x, x) + self.jitter
            self._D = x.reshape(1, -1)
            self._K_inv = np.array([[1.0 / k_xx]], dtype=float)
            return True, k_xx, None

        # Compute k_D(x)
        k_vec = self.kernel.pairwise(self.D, x.reshape(1, -1))[:, 0]  # (m,)
        K_inv = self.K_inv
        a = K_inv @ k_vec  # (m,)
        k_xx = self.kernel(x, x) + self.jitter
        delta = k_xx - (k_vec @ a)

        if delta <= self.nu and compress:
            return False, delta, a

        m = self.size
        s = max(delta, self.jitter)

        K_inv_new = np.empty((m + 1, m + 1), dtype=float)
        K_inv_new[:m, :m] = K_inv + np.outer(a, a) / s
        K_inv_new[:m, m] = -a / s
        K_inv_new[m, :m] = -a / s
        K_inv_new[m, m] = 1.0 / s
        K_inv_new = 0.5 * (K_inv_new + K_inv_new.T)

        self._D = np.vstack([self.D, x.reshape(1, -1)])
        self._K_inv = K_inv_new
        return True, delta, a

    def compute_feature_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.size == 0:
            return np.zeros((0, X.shape[0]), dtype=float)
        KDX = self.kernel.pairwise(self.D, X)  # (m,T)
        return KDX

    def prune(self) -> int:
        diag_vals = np.diag(self._K_inv)
        idx_to_remove = np.argmax(diag_vals)

        s = self._K_inv[idx_to_remove, idx_to_remove]
        keep_mask = np.ones(self.size, dtype=bool)
        keep_mask[idx_to_remove] = False
        v = self._K_inv[keep_mask, idx_to_remove]
        K_inv_sub = self._K_inv[np.ix_(keep_mask, keep_mask)]
        self._K_inv = K_inv_sub - np.outer(v, v) / s
        self._K_inv = 0.5 * (self._K_inv + self._K_inv.T)

        self._D = self._D[keep_mask]

        return idx_to_remove
