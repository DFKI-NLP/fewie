from typing import Union

import numpy
import torch


class Classifier:
    def __call__(
        self,
        X_train: Union[numpy.ndarray, torch.Tensor],
        y_train: Union[numpy.ndarray, torch.Tensor],
        X_test: Union[numpy.ndarray, torch.Tensor],
    ) -> numpy.ndarray:
        pass
