import random
from functools import partial

import numpy
import torch
from sklearn import metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_metric(metric: str):
    return {
        "accuracy": metrics.accuracy_score,
        "precision_micro": partial(metrics.precision_score, average="micro"),
        "recall_micro": partial(metrics.recall_score, average="micro"),
        "f1_micro": partial(metrics.f1_score, average="micro"),
        "f1_macro": partial(metrics.f1_score, average="micro"),
    }[metric]
