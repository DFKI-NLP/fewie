from typing import Union
import numpy
import torch
from fewie.evaluation.classifiers.classifier import Classifier
from sklearn.neighbors import NearestCentroid as NC


def nearest_neighbor(X, neighbors: numpy.ndarray):
    pass


class NearestCentroid(Classifier):
    def __init__(self, metric: str = "euclidean"):
        self.metric = metric

    def __call__(
        self,
        X_train: Union[numpy.ndarray, torch.Tensor],
        y_train: Union[numpy.ndarray, torch.Tensor],
        X_test: Union[numpy.ndarray, torch.Tensor],
    ) -> numpy.ndarray:
        clf = NC(metric=self.metric)
        clf.fit(X_train, y_train)
        return clf.predict(X_test)


class NearestInstance(Classifier):
    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    def __call__(
        self,
        X_train: Union[numpy.ndarray, torch.Tensor],
        y_train: Union[numpy.ndarray, torch.Tensor],
        X_test: Union[numpy.ndarray, torch.Tensor],
    ) -> numpy.ndarray:
        clf = NC(metric="euclidean")
        clf.fit(X_train, y_train)
        return clf.predict(X_test)