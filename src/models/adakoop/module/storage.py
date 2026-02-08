from typing import Any, Optional

import numpy as np

from src.models.adakoop.module.ald import ALDDictionary
from src.models.adakoop.module.lks import LKS


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

        self.lkses = []

    def __call__(self) -> list[LKS]:
        return self.lkses

    def __getitem__(self, idx: int) -> LKS:
        return self.lkses[idx]

    def __len__(self) -> int:
        return len(self.lkses)

    def __iter__(self) -> 'Storage':
        self._i = 0
        return self

    def __next__(self) -> LKS:
        if self._i == len(self):
            raise StopIteration
        lks = self.lkses[self._i]
        self._i += 1
        return lks

    def create(
        self,
        Xc: np.ndarray,
        kernel: Any,
        idx: Optional[int] = None,
        append: bool = False,
    ) -> LKS:
        if idx is None:
            idx = len(self)
        ald = ALDDictionary(
            kernel=kernel,
            nu=self.nu,
            m_max=self.m_max,
            jitter=self.jitter,
        )
        lks = LKS(
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
        lks.fit(data=Xc, compress=self.compress)

        if append:
            self.append(lks)
        return lks, 0

    def append(self, lks: LKS) -> None:
        self.lkses.append(lks)

    def pop(self):
        return self.lkses.pop()
