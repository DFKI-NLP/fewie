import random
from functools import partial

import numpy
import torch
from sklearn import metrics
from typing import Callable


def seed_everything(seed: int) -> None:
    """Sets random seed anywhere randomness is involved.

    This process makes sure all the randomness-involved operations yield the
    same result under the same `seed`, so each experiment is reproducible.
    In this function, we set the same random seed for the following modules:
    `random`, `numpy` and `torch`.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_metric(metric: str) -> Callable:
    """Gets the required metric-function from all supported by `sklearn.metrics`

    Args:
        metric: The name of metric we want, should be one of :
            "accuracy", "precision_micro", "recall_micro", "f1_micro", "f1_macro".

    Returns:
        The corresponding metric-function.
    """
    return {
        "accuracy": metrics.accuracy_score,
        "f1_micro": partial(
            metrics.f1_score, labels=list(range(1, 9)), average="micro"
        ),
        "f1_macro": partial(
            metrics.f1_score, labels=list(range(1, 9)), average="macro"
        ),
    }[metric]
