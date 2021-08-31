from typing import Union
import numpy
import torch
from fewie.evaluation.classifiers.classifier import Classifier
from sklearn.neighbors import NearestCentroid as NC
from sklearn.neighbors import KNeighborsClassifier as KNN


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
    def __init__(
        self,
        n_neighbors: int = 1,
        weights: str = "uniform",
        p: int = 2,
        metric: str = "minkowski",
        n_jobs: int = None,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.metric = metric
        self.n_jobs = n_jobs

    def __call__(
        self,
        X_train: Union[numpy.ndarray, torch.Tensor],
        y_train: Union[numpy.ndarray, torch.Tensor],
        X_test: Union[numpy.ndarray, torch.Tensor],
    ) -> numpy.ndarray:
        clf = KNN(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            p=self.p,
            metric=self.metric,
            n_jobs=None,
        )
        clf.fit(X_train, y_train)
        return clf.predict(X_test)
