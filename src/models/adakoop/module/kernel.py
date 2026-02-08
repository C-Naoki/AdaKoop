from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import (
    euclidean_distances,
    laplacian_kernel,
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)


def build_kernel(data: np.ndarray, kernel_type: str = 'rbf') -> Any:
    if kernel_type == 'rbf':
        lengthscale = np.quantile(pdist(data), 0.5)
        return RBFKernel(lengthscale=lengthscale)
    elif kernel_type == 'poly':
        gamma = 1.0 / float(data.shape[1])
        return PolyKernel(degree=3, gamma=gamma, coef0=1.0)
    elif kernel_type == 'laplace':
        lengthscale = np.quantile(pdist(data, metric='cityblock'), 0.5)
        return LaplacianKernel(lengthscale=lengthscale)
    elif kernel_type == 'exponential':
        lengthscale = np.quantile(pdist(data), 0.5)
        return ExponentialKernel(lengthscale=lengthscale)
    elif kernel_type == 'sigmoid':
        gamma = 1.0 / float(data.shape[1])
        coef0 = 0.0
        return SigmoidKernel(gamma=gamma, coef0=coef0)
    elif kernel_type == 'linear':
        return LinearKernel()
    else:
        raise ValueError(f'Unsupported kernel_type: {kernel_type}')


@dataclass(frozen=True)
class RBFKernel:
    lengthscale: float

    def __post_init__(self) -> None:
        if not (self.lengthscale > 0.0):
            raise ValueError('lengthscale must be positive')

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        diff = x - y
        l2 = self.lengthscale * self.lengthscale
        return np.exp(-0.5 * (diff @ diff) / l2)

    def pairwise(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        gamma = 1.0 / (2.0 * self.lengthscale**2)
        return rbf_kernel(X, Y, gamma=gamma)


@dataclass(frozen=True)
class LaplacianKernel:
    lengthscale: float

    def __post_init__(self) -> None:
        if not (self.lengthscale > 0.0):
            raise ValueError('lengthscale must be positive')

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        d1 = np.sum(np.abs(x - y))
        return np.exp(-d1 / self.lengthscale)

    def pairwise(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        gamma = 1.0 / self.lengthscale
        return laplacian_kernel(X, Y, gamma=gamma)


@dataclass(frozen=True)
class ExponentialKernel:
    lengthscale: float

    def __post_init__(self) -> None:
        if not (self.lengthscale > 0.0):
            raise ValueError('lengthscale must be positive')

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        dist = np.linalg.norm(x - y)
        return np.exp(-dist / self.lengthscale)

    def pairwise(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        dists = euclidean_distances(X, Y)
        return np.exp(-dists / self.lengthscale)


@dataclass(frozen=True)
class PolyKernel:
    degree: int = 3
    gamma: Optional[float] = None
    coef0: float = 1.0

    def __post_init__(self) -> None:
        if not (isinstance(self.degree, int) and self.degree >= 0):
            raise ValueError('degree must be a non-negative integer')
        if self.gamma is not None and not (self.gamma > 0.0):
            raise ValueError('gamma must be positive when provided')

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        gamma = self.gamma if self.gamma is not None else (1.0 / x.size)
        return (gamma * (x @ y) + self.coef0) ** self.degree

    def pairwise(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        gamma = self.gamma if self.gamma is not None else (1.0 / X.shape[1])
        return polynomial_kernel(X, Y, degree=self.degree, gamma=gamma, coef0=self.coef0)


@dataclass(frozen=True)
class SigmoidKernel:
    gamma: float
    coef0: float

    def __post_init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        dot_prod = x @ y
        return np.tanh(self.gamma * dot_prod + self.coef0)

    def pairwise(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return sigmoid_kernel(X, Y, gamma=self.gamma, coef0=self.coef0)


@dataclass(frozen=True)
class LinearKernel:
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return x @ y

    def pairwise(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return linear_kernel(X, Y)
