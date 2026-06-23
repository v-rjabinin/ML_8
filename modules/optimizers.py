from additional_descriptors import PositiveInt, Ratio
from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    @abstractmethod
    def step(self, grads: list[np.ndarray] | tuple[np.ndarray, ...]):
        """Each optimizer must define how it updates parameters"""

    @abstractmethod
    def epoch_step(self, n_epoch: int):
        """Each optimizer must define how it behaves at the end of the epoch"""


class SGD(BaseOptimizer):
    lr, gamma = Ratio(), Ratio()
    step_size = PositiveInt()

    def __init__(self, params: list[np.ndarray] | tuple[np.ndarray, ...], lr: float = 0.01, gamma: float = 0.1, step_size: int = 10):
        if any(not isinstance(arr, np.ndarray) for arr in params):
            raise TypeError("'params' must only contain numpy.ndarrays")
        if any(not np.issubdtype(arr.dtype, np.floating) for arr in params):
            raise TypeError("elements in 'params' must be subdtypes of numpy.floating")

        self.step_size = step_size
        self.params = params
        self.gamma = gamma
        self.lr = lr

    def step(self, grads: list[np.ndarray] | tuple[np.ndarray, ...]):
        params = self.params
        if len(grads) != len(params):
            raise ValueError(f"length of 'grads' must be equal to the length of 'params'")

        l = len(grads)

        if any(grads[i].shape != params[i].shape for i in range(l)):
            raise ValueError(f"arrays in 'grads' must have the same shapes as the arrays in 'params'")

        for i in range(l):
            self.params[i] -= self.lr * grads[i]

    def epoch_step(self, n_epoch: int) -> None:
        if (n_epoch + 1) % self.step_size == 0:
            self.lr *= self.gamma


class Adam(BaseOptimizer):
    lr, p1, p2 = Ratio(), Ratio(), Ratio()

    def __init__(self, params: list[np.ndarray] | tuple[np.ndarray, ...], lr: float = 0.01, p1: float = 0.9, p2: float = 0.99, eps: float = 1e-8):
        if any(not isinstance(arr, np.ndarray) for arr in params):
            raise TypeError("'params' must only contain numpy.ndarrays")
        if any(not np.issubdtype(arr.dtype, np.floating) for arr in params):
            raise TypeError("elements in 'params' must be subdtypes of numpy.floating")

        self.params = params
        self.lr, self.eps = lr, eps
        self.p1, self.p2 = p1, p2

        l = len(params)
        self.s = [np.zeros_like(params[i]) for i in range(l)]
        self.r = [np.zeros_like(params[i]) for i in range(l)]
        self.t = 0

    def step(self, grads: list[np.ndarray] | tuple[np.ndarray, ...]):
        params = self.params
        if len(grads) != len(params):
            raise ValueError(f"length of 'grads' must be equal to the length of 'params'")

        l = len(grads)

        if any(grads[i].shape != params[i].shape for i in range(l)):
            raise ValueError(f"arrays in 'grads' must have the same shapes as the arrays in 'params'")

        self.t += 1
        t, lr, p1, p2, s, r, eps = self.t, self.lr, self.p1, self.p2, self.s, self.r, self.eps

        for i in range(l):
            s[i] = p1 * s[i] + (1 - p1) * grads[i]
            r[i] = p2 * r[i] + (1 - p2) * (grads[i] * grads[i])
            s_norm, r_norm = s[i] / (1 - p1 ** t), r[i] / (1 - p2 ** t)
            params[i] -= lr * (s_norm / (np.sqrt(r_norm) + eps))

    def epoch_step(self, n_epoch: int) -> None:
        pass


if __name__ == "__main__":
    print("Module containing SGD/Adam optimizers imported✅")