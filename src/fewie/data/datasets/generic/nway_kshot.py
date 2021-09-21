import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import datasets

logger = logging.getLogger(__name__)


class NwayKshotDataset(torch.utils.data.Dataset):
    """Few-shot (with `N` ways and `K` shots) dataset sampled from the \
        original dataset.
    """

    def __init__(
        self,
        dataset: datasets.Dataset,
        n_ways: int = 5,
        k_shots: int = 1,
        n_queries: int = 1,
        n_samples: int = 600,
        label_column_name: str = "ner_tags",
        columns: Optional[List[str]] = None,
        ignore_labels: Optional[List[str]] = None,
        deterministic: bool = False,
    ):
        """
        Args:
            dataset: The originald dataset from which we generate few-shot dataset.
            n_ways: `N`, the number of classes in the few-shot dataset.
            k_shots: `K`, the number of samples in the support set for each class.
            n_queries: `K'`, The number of samples in the query set for each class.
            n_samples: Length of dataset, which equals the number of episodes.
            label_column_name: The name of NER-column, e.g. "ner-tags" for most cases.
            columns: The names of the columns we still want to keep, not relevant for now.
            ignore_labels: The list of NER-classes to be ignored, i.e. too under-represented \
                that there are not even enough samples from the class(es).
            deterministic: Whether to reset random seed for batcher, not relevant for now.
        """
        super().__init__()
        self.n_ways = n_ways
        self.k_shots = k_shots
        self.n_queries = n_queries
        self.n_samples = n_samples
        self.dataset = dataset
        self.columns = columns
        self.deterministic = deterministic

        self.feature = self.dataset.features[label_column_name].feature
        # get all the possible classes (i.e. entity types)
        classes = set(self.feature._str2int.values())
        if ignore_labels is not None:
            classes -= set(self.feature.str2int(ignore_labels))
        class_labels = list(classes)

        # For each class, maintain a list of text-ids where entities from it can be
        # found. Notice that if the i-th text contains n tokens from class C (given by
        # class id: c), then `i` appears in list `class_indices[c]` n times.
        self.class_indices: Dict[int, List[int]] = {}
        for idx, example in enumerate(dataset):
            for label in example[label_column_name]:
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

        # oroginal class-ids after ignoring overly under-represented classes
        self.classes = sorted(list(self.class_indices.keys()))

        logger.info(
            "The following classes have an insufficient number of examples "
            "for the current setting: {}".format(self.feature.int2str(ignored_classes))
        )

        logger.info(
            "Num examples: {}".format(
                {self.feature.int2str(k): len(v) for k, v in self.class_indices.items()}
            )
        )

    def _sample_classes(self) -> np.ndarray:
        """Sample `N` classes from all the existing classes.

        Returns: a N-length list of original class-id with sufficient samples.
        """
        return np.random.choice(self.classes, self.n_ways, replace=False)

    def _sample_indices_and_targets(
        self, cls_sampled: np.ndarray
    ) -> Tuple[
        List[np.ndarray],
        List[List[int]],
        List[List[int]],
        List[np.ndarray],
        List[List[int]],
        List[List[int]],
    ]:
        """Samples a support set of size K and query set of size K'.

        Args:
            cls_sampled: A list of length N consisting of the sampled class-ids.

        Returns:
            support_indices: A list of text-ids for support set, of length N*K.
            support_targets: A list of encoded class-ids for support set.
            support_targets_orig: A list of original class-ids for support set.
            query_indices: A list of text-ids for query set, of length N*K'.
            query_targets: A list of encoded class-ids for query set.
            query_targets_orig: A list of original class-ids for query set.
        """
        support_indices = []
        support_targets = []
        query_indices = []
        query_targets = []
        for idx, cls in enumerate(cls_sampled):
            cls_indices = np.asarray(self.class_indices[cls])

            support_ids = np.random.choice(
                range(cls_indices.shape[0]), self.k_shots, replace=False
            )
            support_indices.append(cls_indices[support_ids])
            support_targets.append([cls] * self.k_shots)

            # make sure support and query have no overlap for each sampled class
            query_ids = np.setxor1d(np.arange(cls_indices.shape[0]), support_ids)
            query_ids = np.random.choice(query_ids, self.n_queries, replace=False)
            query_indices.append(cls_indices[query_ids])
            query_targets.append([cls] * self.n_queries)

        return (
            support_indices,
            support_targets,
            query_indices,
            query_targets,
        )

    def __getitem__(self, item):
        """Fetches a batch of data from original dataset using the sampled indices.

        Args:
            item: The batch id, not relevant if we stick to for-loop to iterate the dataloader.

        Returns:
            support: A list of texts for support set, of shape `[N*K, seq_len]`.
            support_targets: A numpy list of encoded class-ids for support set.
            support_targets_orig: A numpu list of original class-ids for support set.
            query_indices: A list of texts for query set, of shape `[N*K', seq_len]`.
            query_targets: A numpy list of encoded class-ids for query set.
            query_targets_orig: A numpy list of original class-ids for query set.
        """
        if self.deterministic:
            np.random.seed(item)

        cls_sampled = self._sample_classes()

        (
            support_indices,
            support_targets,
            query_indices,
            query_targets,
        ) = self._sample_indices_and_targets(cls_sampled)

        # To make the indices work for fetching data from the original dataset, we need to
        # convert the native List[int] into numpy array.
        support_indices = np.concatenate(support_indices).flatten()
        support_targets = np.concatenate(support_targets).flatten()

        query_indices = np.concatenate(query_indices).flatten()
        query_targets = np.concatenate(query_targets).flatten()

        with self.dataset.formatted_as(type="numpy", columns=self.columns):
            support = self.dataset[support_indices]
            query = self.dataset[query_indices]

        return (
            support,
            support_targets,
            query,
            query_targets,
        )

    def __len__(self):
        """The number of episodes (i.e. runs) of the few-shot dataset."""
        return self.n_samples


class NwayKshotNaDedicatedDataset(NwayKshotDataset):
    """Few-shot (with `N` ways and `K` shots) dataset sampled from the original \
        dataset, which supports users adding an 'NA'-class using a customizable \
        class name, and mandatorily contains this class in the sampled classes.
    """

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
        label_column_name: str = "label",
        deterministic: bool = False,
    ):
        """
        Args:
            dataset: The originald dataset from which we generate few-shot dataset.
            n_ways: `N`, the number of classes in the few-shot dataset.
            k_shots: `K`, the number of samples in the support set for each class.
            n_queries: `K'`, The number of samples in the query set for each class.
            n_samples: Length of dataset, which equals the number of episodes.
            na_label: The user-defined tag name for 'NA'-class.
            columns: The names of the columns we still want to keep, not relevant for now.
            ignore_labels: The list of NER-classes to be ignored, i.e. too under-represented \
                that there are not even enough samples from the class(es).
            deterministic: Whether to reset random seed for batcher, not relevant for now.
        """
        super().__init__(
            dataset=dataset,
            n_ways=n_ways,
            k_shots=k_shots,
            n_queries=n_queries,
            n_samples=n_samples,
            columns=columns,
            ignore_labels=ignore_labels,
            label_column_name=label_column_name,
            deterministic=deterministic,
        )
        self.na_label_idx = self.class_str2int(na_label)

    def _sample_classes(self) -> np.ndarray:
        sampled_classes = np.random.choice(self.classes, self.n_ways, replace=False)
        if self.na_label_idx not in sampled_classes:
            sampled_classes[0] = self.na_label_idx
        return sampled_classes


class NwayKshotNaRestDataset(NwayKshotDataset):
    """Few-shot (with `N` ways and `K` shots) dataset sampled from the original \
        dataset, which mandatorily samples N-1 classes first, then treat all the \
        rest classes as one class ("negative") to be sampled and added.
    """

    def _sample_classes(self) -> np.ndarray:
        """Sample N-1 classes from all the possible classes in the original dataset."""
        return np.random.choice(self.classes, self.n_ways - 1, replace=False)

    def _sample_indices_and_targets(
        self, cls_sampled: np.ndarray
    ) -> Tuple[List[np.ndarray], List[List[int]], List[np.ndarray], List[List[int]]]:
        """Sample support set of size K and query set of size K'.

        Args:
            cls_sampled: A list of length N-1 consisting of the sampled class-ids.

        Returns:
            support_indices: A list of text-ids for support set, of length N*K.
            support_targets: A list of encoded class-ids for support set.
            query_indices: A list of text-ids for query set, of length N*K'.
            query_targets: A list of encoded class-ids for query set.
        """
        support_indices = []
        support_targets = []
        query_indices = []
        query_targets = []
        for idx, cls in enumerate(cls_sampled):
            cls_indices = np.asarray(self.class_indices[cls])

            support_ids = np.random.choice(
                range(cls_indices.shape[0]), self.k_shots, replace=False
            )
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

            support_ids = np.random.choice(
                range(cls_indices.shape[0]), count, replace=False
            )
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

            query_ids = np.random.choice(
                range(cls_indices.shape[0]), count, replace=False
            )
            other_query_indices.extend(cls_indices[query_ids])
            other_query_targets.extend([self.n_ways] * count)

        query_indices.append(other_query_indices)
        query_targets.append(other_query_targets)

        return support_indices, support_targets, query_indices, query_targets


class ContrastiveNwayKshotDataset(NwayKshotDataset):
    def _sample_indices_and_targets(
        self, cls_sampled: np.ndarray
    ) -> Tuple[
        List[np.ndarray],
        List[List[int]],
        List[List[int]],
        List[np.ndarray],
        List[List[int]],
        List[List[int]],
        List[np.ndarray],
        List[List[int]],
        List[List[int]],
    ]:
        contrastive_indices, contrastive_targets, contrastive_targets_orig = [], [], []
        support_indices, support_targets = [], []
        query_indices, query_targets = [], []

        for idx, cls in enumerate(cls_sampled):
            cls_indices = np.asarray(self.class_indices[cls])

            # build positive pairs
            positive_ids_pool = np.random.choice(
                range(cls_indices.shape[0]), max(self.k_shots, 2), replace=False
            )
            positive_ids = np.random.choice(positive_ids_pool, 2)
            contrastive_indices.append(cls_indices[positive_ids])
            contrastive_targets.append([0])
            contrastive_targets_orig.append([cls, cls])

            # sample support and query set as base class does
            support_ids = positive_ids_pool[: self.k_shots]
            support_indices.append(cls_indices[support_ids])
            support_targets.append([cls] * self.k_shots)

            query_ids = np.setxor1d(np.arange(cls_indices.shape[0]), support_ids)
            query_ids = np.random.choice(query_ids, self.n_queries, replace=False)
            query_indices.append(cls_indices[query_ids])
            query_targets.append([cls] * self.n_queries)

        # build negative pairs
        for idx, cls in enumerate(cls_sampled):
            negative_pool = self.class_indices.copy()
            negative_pool.pop(cls)
            for _ in range(self.k_shots):
                negative_cls = random.choice(list(negative_pool.keys()))
                negative_id = random.choice(negative_pool[negative_cls])
                contrastive_indices.append([contrastive_indices[idx][0], negative_id])
                contrastive_targets.append([1])
                contrastive_targets_orig.append([cls, negative_cls])

        return (
            contrastive_indices,
            contrastive_targets,
            contrastive_targets_orig,
            support_indices,
            support_targets,
            query_indices,
            query_targets,
        )

    def __getitem__(self, item):
        if self.deterministic:
            np.random.seed(item)

        cls_sampled = self._sample_classes()

        (
            contrastive_indices,
            contrastive_targets,
            contrastive_targets_orig,
            support_indices,
            support_targets,
            query_indices,
            query_targets,
        ) = self._sample_indices_and_targets(cls_sampled)

        contrastive_left_indices = [
            contrastive_pair[0] for contrastive_pair in contrastive_indices
        ]
        contrastive_right_indices = [
            contrastive_pair[1] for contrastive_pair in contrastive_indices
        ]
        contrastive_targets = np.concatenate(contrastive_targets).flatten()
        contrastive_targets_orig = np.array(contrastive_targets_orig)

        support_indices = np.concatenate(support_indices).flatten()
        support_targets = np.concatenate(support_targets).flatten()

        query_indices = np.concatenate(query_indices).flatten()
        query_targets = np.concatenate(query_targets).flatten()

        with self.dataset.formatted_as(type="numpy", columns=self.columns):
            contrastive_left = self.dataset[contrastive_left_indices]
            contrastive_right = self.dataset[contrastive_right_indices]
            support = self.dataset[support_indices]
            query = self.dataset[query_indices]

        contrastive = (contrastive_left, contrastive_right)
        return (
            contrastive,
            contrastive_targets,
            contrastive_targets_orig,
            support,
            support_targets,
            query,
            query_targets,
        )
