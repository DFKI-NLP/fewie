import logging
from typing import Dict, List, Optional, Tuple

import datasets
import numpy as np
import torch


logger = logging.getLogger(__name__)


class NwayKshotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        n_ways: int,
        k_shots: int,
        n_queries: int,
        n_samples: int,
        columns: Optional[List[str]] = None,
        ignore_labels: Optional[List[str]] = None,
        label_column: str = "label",
        deterministic: bool = False,
    ):
        super().__init__()
        self.n_ways = n_ways
        self.k_shots = k_shots
        self.n_queries = n_queries
        self.n_samples = n_samples
        self.dataset = dataset
        self.columns = columns
        self.deterministic = deterministic

        self.class_int2str = self.dataset.features[label_column].int2str
        self.class_str2int = self.dataset.features[label_column].str2int

        classes = set(dataset[label_column])
        if ignore_labels is not None:
            classes -= set(self.class_str2int(ignore_labels))
        class_labels = list(classes)

        self.class_indices: Dict[int, List[int]] = {}
        for idx, example in enumerate(dataset):
            label = example[label_column]

            if label not in class_labels:
                continue

            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        ignored_classes = []
        min_num_examples = n_ways * k_shots + n_ways * n_queries
        for class_name, class_ids in self.class_indices.items():
            num_examples = len(class_ids)
            if num_examples < min_num_examples:
                ignored_classes.append(class_name)

        for key in ignored_classes:
            self.class_indices.pop(key, None)

        self.classes = list(self.class_indices.keys())

        logger.info(
            "The following classes have an insufficient number of examples for the current setting: %s"
            % self.class_int2str(ignored_classes)
        )

        logger.info(
            "Num examples: %s"
            % {self.class_int2str(k): len(v) for k, v in self.class_indices.items()}
        )

    def _sample_classes(self) -> np.ndarray:
        return np.random.choice(self.classes, self.n_ways, replace=False)

    def _sample_indices_and_targets(
        self, cls_sampled: np.ndarray
    ) -> Tuple[List[np.ndarray], List[List[int]], List[np.ndarray], List[List[int]]]:
        support_indices = []
        support_targets = []
        query_indices = []
        query_targets = []
        for idx, cls in enumerate(cls_sampled):
            cls_indices = np.asarray(self.class_indices[cls])

            support_ids = np.random.choice(range(cls_indices.shape[0]), self.k_shots, replace=False)
            support_indices.append(cls_indices[support_ids])
            support_targets.append([idx] * self.k_shots)

            query_ids = np.setxor1d(np.arange(cls_indices.shape[0]), support_ids)
            query_ids = np.random.choice(query_ids, self.n_queries, replace=False)
            query_indices.append(cls_indices[query_ids])
            query_targets.append([idx] * self.n_queries)

        return support_indices, support_targets, query_indices, query_targets

    def __getitem__(self, item):
        if self.deterministic:
            np.random.seed(item)

        cls_sampled = self._sample_classes()

        (
            support_indices,
            support_targets,
            query_indices,
            query_targets,
        ) = self._sample_indices_and_targets(cls_sampled)

        support_indices = np.concatenate(support_indices).flatten()
        support_targets = np.concatenate(support_targets).flatten()
        query_indices = np.concatenate(query_indices).flatten()
        query_targets = np.concatenate(query_targets).flatten()

        with self.dataset.formatted_as(type="numpy", columns=self.columns):
            support = self.dataset[support_indices]
            query = self.dataset[query_indices]

        return support, support_targets, query, query_targets

    def __len__(self):
        return self.n_samples


class NwayKshotNaDedicatedDataset(NwayKshotDataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        n_ways: int,
        k_shots: int,
        n_queries: int,
        n_samples: int,
        na_label: str,
        columns: Optional[List[str]] = None,
        ignore_labels: Optional[List[str]] = None,
        label_column: str = "label",
        deterministic: bool = False,
    ):
        super().__init__(
            dataset=dataset,
            n_ways=n_ways,
            k_shots=k_shots,
            n_queries=n_queries,
            n_samples=n_samples,
            columns=columns,
            ignore_labels=ignore_labels,
            label_column=label_column,
            deterministic=deterministic,
        )
        self.na_label_idx = self.class_str2int(na_label)

    def _sample_classes(self) -> np.ndarray:
        sampled_classes = np.random.choice(self.classes, self.n_ways, replace=False)
        if self.na_label_idx not in sampled_classes:
            sampled_classes[0] = self.na_label_idx
        return sampled_classes


class NwayKshotNaRestDataset(NwayKshotDataset):
    def _sample_classes(self) -> np.ndarray:
        return np.random.choice(self.classes, self.n_ways - 1, replace=False)

    def _sample_indices_and_targets(
        self, cls_sampled: np.ndarray
    ) -> Tuple[List[np.ndarray], List[List[int]], List[np.ndarray], List[List[int]]]:
        support_indices = []
        support_targets = []
        query_indices = []
        query_targets = []
        for idx, cls in enumerate(cls_sampled):
            cls_indices = np.asarray(self.class_indices[cls])

            support_ids = np.random.choice(range(cls_indices.shape[0]), self.k_shots, replace=False)
            support_indices.append(cls_indices[support_ids])
            support_targets.append([idx] * self.k_shots)

            query_ids = np.setxor1d(np.arange(cls_indices.shape[0]), support_ids)
            query_ids = np.random.choice(query_ids, self.n_queries, replace=False)
            query_indices.append(cls_indices[query_ids])
            query_targets.append([idx] * self.n_queries)

        # sample negative instances from the remaining classes
        other_cls_ids = np.setxor1d(self.classes, cls_sampled)
        other_classes_sampled = np.random.choice(
            other_cls_ids, self.k_shots + self.n_queries, replace=True
        )

        # sample negatives for support set
        other_support_indices = []
        other_support_targets = []
        for cls, count in zip(
            *np.unique(other_classes_sampled[: self.k_shots], return_counts=True)
        ):
            cls_indices = np.asarray(self.class_indices[cls])

            support_ids = np.random.choice(range(cls_indices.shape[0]), count, replace=False)
            other_support_indices.extend(cls_indices[support_ids])
            other_support_targets.extend([self.n_ways] * count)

        support_indices.append(other_support_indices)
        support_targets.append(other_support_targets)

        # sample negatives for query set
        other_query_indices = []
        other_query_targets = []
        for cls, count in zip(
            *np.unique(other_classes_sampled[self.k_shots :], return_counts=True)
        ):
            cls_indices = np.asarray(self.class_indices[cls])

            support_ids = np.random.choice(range(cls_indices.shape[0]), count, replace=False)
            other_query_indices.extend(cls_indices[support_ids])
            other_query_targets.extend([self.n_ways] * count)

        query_indices.append(other_query_indices)
        query_targets.append(other_query_targets)

        return support_indices, support_targets, query_indices, query_targets
