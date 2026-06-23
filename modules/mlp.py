from additional_descriptors import PositiveInt, NonNegativeInt, Ratio
from optimizers import BaseOptimizer, Adam
from sklearn.utils import check_array, check_X_y
from typing import Callable, Type
from tqdm import tqdm
import pandas as pd
import numpy as np
import time


class MLP:
    thld, hidden_dim, random_state = Ratio(), PositiveInt(), NonNegativeInt()
    valid_labels = {0, 1}
    sig = lambda obj, z: 1 / (1 + np.exp(-z))

    def __init__(self, n_hidden: int, activation: Callable[[np.ndarray], np.ndarray], thld: float = 0.5, random_state: int = 21):
        self.hidden_dim = n_hidden
        self.act_fc = activation
        self.thld = thld

        self.random_state = random_state
        self._n_features = -1

        self._weights = None
        self._biases = None

    def fit(self, x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, batch_size: int = 32, n_epoch: int = 20, optim_class: Type[BaseOptimizer] = Adam, optim_params: dict = {}, sleep_step: int = 3):
        x, y = self._validate_x_y(x, y)
        self._validate_pos_int(batch_size, "batch_size")
        self._validate_pos_int(n_epoch, "n_epoch")

        n_features, hidden_dim = x.shape[1], self.hidden_dim

        self._n_features = n_features

        if not set(np.unique(y)).issubset(self.valid_labels):
            raise ValueError(f"'y' must only contain labels from {self.valid_labels}")

        rng = np.random.default_rng(seed=self.random_state)

        W1 = rng.normal(scale=0.01, size=(n_features, hidden_dim))
        W2 = rng.normal(scale=0.01, size=(hidden_dim, 1))
        b1 = rng.normal(scale=0.01, size=hidden_dim)
        b2 = rng.normal(scale=0.01, size=1)

        optimizer = optim_class([W1, W2, b1, b2], **optim_params)

        weights = [W1, W2]
        biases = [b1, b2]

        states = rng.integers(low=0, high=2 ** 16 - 1, size=n_epoch, dtype=np.int32)

        for epoch in range(n_epoch):
            idx_gen = self._random_idx(x.shape[0], batch_size, int(states[epoch]))
            loss_mean = 0.0
            lm_count = 0

            train_tqdm = tqdm(idx_gen, leave=True)
            for curr_idx in train_tqdm:
                curr_batch, y_true = x[curr_idx, :], y[curr_idx]

                interim_values = self._forward_prop(weights, biases, curr_batch)
                grads = self._backprop(interim_values, curr_batch, y_true, weights)
                optimizer.step(grads)

                lm_count += 1
                p, var_y = interim_values[1][1], y_true.reshape(-1, 1)
                p = np.clip(p, 1e-15, 1 - 1e-15)
                loss = -(var_y * np.log(p) + (1 - var_y) * np.log(1 - p))
                loss = np.mean(loss, axis=0)
                loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
                train_tqdm.set_description(f"Epoch [{epoch + 1}/{n_epoch}], loss_mean={loss_mean:.3f}")

            optimizer.epoch_step(epoch)

            if (epoch + 1) % sleep_step == 0: ## required to prevent errors in Jupyter Notebook; can be removed
                time.sleep(2.0)

        self._weights = weights
        self._biases = biases

    def _forward_prop(self, weights: list[np.ndarray], biases: list[np.ndarray], x: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        act_fc, sig = self.act_fc, self.sig

        # x - R^(l x d), W1 - R^(d x hidd_dim), W2 - R^(hidd_dim x 1), b1 - R^(hidd_dim), b2 - R^(1)
        z1 = x @ weights[0] + biases[0]     # z1 - (l x hidd_dim)
        f1 = act_fc(z1)                     # f1 - (l x hidd_dim)

        z2 = f1 @ weights[1] + biases[1]    # z2 - (l x 1)
        f2 = sig(z2)                        # f2 - (l x 1)

        return [(z1, z2), (f1, f2)]

    def _backprop(self, interim_results: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray, y: np.ndarray, weights: list[np.ndarray]) -> list[np.ndarray]:
        (z1, z2), (f1, f2) = interim_results
        y = y.reshape(-1, 1)

        dL_dz2 = f2 - y                                             # (l x 1)

        dL_dW2 = f1.T @ dL_dz2                                      # (hidd_dim x 1)
        dL_db2 = np.sum(dL_dz2, axis=0)                             # (1, )

        dL_df1 = dL_dz2 @ weights[1].reshape(1, -1)                 # (l x hidd_dim)
        dL_dz1 = dL_df1 * self._calc_deriv(self.act_fc, z1)         # (l x hidd_dim)

        dL_dW1 = x.T @ dL_dz1                                       # (d x hidd_dim)
        dL_db1 = np.sum(dL_dz1, axis=0)                             # (hidd_dim, )

        batch_size = x.shape[0]
        return [dL_dW1 / batch_size, dL_dW2 / batch_size, dL_db1 / batch_size, dL_db2 / batch_size]

    def predict_proba(self, x: np.ndarray | pd.DataFrame):
        if self._weights is None or self._biases is None:
            raise AttributeError(f"Model must be fitted before making predictions")

        x = self._validate_x(x)

        if x.shape[1] != self._n_features:
            raise ValueError(f"'x' must have the same second dimension as the one seen when fitting")

        probas = self._forward_prop(self._weights, self._biases, x)[1][1]

        return np.hstack([1 - probas, probas])

    def predict(self, x: np.ndarray | pd.DataFrame):
        probas = self.predict_proba(x)[:, 1]

        return (probas > self.thld).astype(int)

    def __repr__(self):
        return f"MLP(n_hidden={self.hidden_dim}, activation={self.act_fc.__name__})"

    @staticmethod
    def _calc_deriv(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return (func(x + eps) - func(x - eps)) / (2 * eps)

    @staticmethod
    def _random_idx(length: int, batch_size: int, random_state: int):
        rng = np.random.default_rng(seed=random_state)
        idx = rng.choice(length, size=length, replace=False)

        start, end = -batch_size, 0
        while end < length:
            start += batch_size
            end = min(end + batch_size, length)

            yield idx[start:end]

    @staticmethod
    def _validate_x(x: np.ndarray | pd.DataFrame, input_name: str = "x") -> np.ndarray:
        x = check_array(x, accept_sparse=['csr'], input_name=input_name)

        return x

    @staticmethod
    def _validate_x_y(x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
        x, y = check_X_y(x, y, accept_sparse=['csr'])

        return x, y

    @staticmethod
    def _validate_pos_int(value: int, arg_name: str):
        try:
            value = int(value)
        except ValueError:
            raise TypeError(f"'{arg_name}' must be an integer")

        if value < 1:
            raise ValueError(f"'{arg_name}' must be positive")

    @property
    def act_fc(self):
        return self._act_fc

    @act_fc.setter
    def act_fc(self, act_fc):
        if not callable(act_fc):
            raise TypeError("Parameter 'act_fc' must be a callable object")

        self._act_fc = act_fc


if __name__ == "__main__":
    print("Module containing MLP classifier imported✅")