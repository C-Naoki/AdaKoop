from typing import Any, Optional

import numpy as np

from src.models.adakoop.module.ald import ALDDictionary
from src.models.adakoop.module.dks import DKS


class Storage:
    def __init__(
        self,
        nu: float = 0.8,
        gamma: float = 0.02,
        lambda_A: float = 1e-6,
        em_iters: int = 3,
        em_tol: float = 1e-6,
        r_init: float = 1e-2,
        m_max: int = 100,
        compress: bool = True,
        jitter: float = 1e-10,
        verbose: bool = True,
    ) -> None:
        self.nu = nu
        self.gamma = gamma
        self.lambda_A = lambda_A
        self.em_iters = em_iters
        self.em_tol = em_tol
        self.r_init = r_init
        self.m_max = m_max
        self.compress = compress
        self.jitter = jitter
        self.verbose = verbose

        self.dks_ls = []

    def __call__(self) -> list[DKS]:
        return self.dks_ls

    def __getitem__(self, idx: int) -> DKS:
        return self.dks_ls[idx]

    def __len__(self) -> int:
        return len(self.dks_ls)

    def __iter__(self) -> 'Storage':
        self._i = 0
        return self

    def __next__(self) -> DKS:
        if self._i == len(self):
            raise StopIteration
        dks = self.dks_ls[self._i]
        self._i += 1
        return dks

    def create(
        self,
        Xc: np.ndarray,
        kernel: Any,
        idx: Optional[int] = None,
        append: bool = False,
    ) -> DKS:
        if idx is None:
            idx = len(self)
        ald = ALDDictionary(
            kernel=kernel,
            nu=self.nu,
            m_max=self.m_max,
            jitter=self.jitter,
        )
        dks = DKS(
            idx=idx,
            ald=ald,
            gamma=self.gamma,
            lambda_A=self.lambda_A,
            em_iters=self.em_iters,
            em_tol=self.em_tol,
            r_init=self.r_init,
            jitter=self.jitter,
            verbose=self.verbose,
        )
        dks.fit(data=Xc, compress=self.compress)

        if append:
            self.append(dks)
        return dks, 0

    def append(self, dks: DKS) -> None:
        self.dks_ls.append(dks)

    def pop(self):
        return self.dks_ls.pop()
