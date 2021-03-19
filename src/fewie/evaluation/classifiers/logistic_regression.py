from typing import Union

import numpy
import torch
from sklearn.linear_model import LogisticRegression as LR


class LogisticRegression:
    def __init__(
        self,
        penalty: str = "l2",
        random_state: int = 0,
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        multi_class: str = "multinomial",
    ):
        self.penalty = penalty
        self.random_state = random_state
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class

    def __call__(
        self,
        X_train: Union[numpy.ndarray, torch.Tensor],
        y_train: Union[numpy.ndarray, torch.Tensor],
        X_test: Union[numpy.ndarray, torch.Tensor],
    ) -> numpy.ndarray:
        clf = LR(
            penalty=self.penalty,
            random_state=self.random_state,
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            multi_class=self.multi_class,
        )
        clf.fit(X_train, y_train)
        return clf.predict(X_test)
