from typing import Dict, List, Optional

import numpy as np
import scipy
import torch
from tqdm import tqdm

import datasets
from fewie.data.datasets.generic.nway_kshot import NwayKshotDataset
from fewie.encoders.encoder import Encoder
from fewie.evaluation.classifiers.classifier import Classifier
from fewie.evaluation.utils import get_metric


def mean_confidence_interval(data: List[float], confidence: float = 0.95):
    array = np.array(data)
    num = len(array)
    ddof = num - 1
    mean, std_error_mean = np.mean(array), scipy.stats.sem(array)
    margin_of_error = std_error_mean * scipy.stats.t._ppf((1.0 + confidence) / 2.0, ddof)
    return (
        mean,
        margin_of_error,
        scipy.stats.t.interval(0.95, ddof, loc=np.mean(array), scale=scipy.stats.sem(array)),
    )


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1.0 / 2)
    out = x.div(norm)
    return out


def prepare_features(
    support_features,
    support_targets,
    support_targets_orig,
    support_labels,
    query_features,
    query_targets,
    query_targets_orig,
    query_labels,
):
    X_support = []
    y_support = []
    for i, (target, target_orig, labels) in enumerate(
        zip(support_targets, support_targets_orig, support_labels)
    ):
        mask = labels == target_orig
        features = support_features[i, mask, :]
        X_support.append(features)
        y_support.extend([target] * features.shape[0])

    X_query = []
    y_query = []
    for i, (target, target_orig, labels) in enumerate(
        zip(query_targets, query_targets_orig, query_labels)
    ):
        mask = labels == target_orig
        features = query_features[i, mask, :]
        X_query.append(features)
        y_query.extend([target] * features.shape[0])

    X_support = np.concatenate(X_support, axis=0)
    y_support = np.array(y_support)
    X_query = np.concatenate(X_query, axis=0)
    y_query = np.array(y_query)

    return X_support, y_support, X_query, y_query


def eval_few_shot_linear_readout(
    encoder: Encoder,
    dataset: datasets.Dataset,
    few_shot_dataset: NwayKshotDataset,
    classifier: Classifier,
    batch_size: int,
    device: torch.device,
    normalize_embeddings: bool = True,
    confidence: float = 0.95,
    ignore_labels: Optional[List[str]] = None,
    deterministic: bool = False,
    metrics: Optional[List[str]] = None,
):
    encoder = encoder.eval()

    dataloader = torch.utils.data.DataLoader(
        few_shot_dataset,
        batch_size=batch_size,
    )

    n_ways = few_shot_dataset.n_ways
    k_shots = few_shot_dataset.k_shots
    n_queries = few_shot_dataset.n_queries

    if metrics is None:
        metrics = ["accuracy"]

    scorers = {metric: get_metric(metric) for metric in metrics}

    metric_scores: Dict[str, List[float]] = {metric: [] for metric in metrics}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # support: [batch_size, n_ways * k_shots, ...]
            # query: [batch_size, n_ways * n_queries, ...]
            # support_targets: [batch_size, n_ways * k_shots]
            # query_targets: [batch_size, n_ways * n_queries]
            (
                support,
                support_targets,
                support_targets_orig,
                query,
                query_targets,
                query_targets_orig,
            ) = batch

            batch_size, _, seq_len = support["input_ids"].shape

            support_labels = support["labels"].cpu().numpy()
            query_labels = query["labels"].cpu().numpy()

            support = {
                key: tensor.to(device).view(batch_size * n_ways * k_shots, seq_len).long()
                for key, tensor in support.items()
                if key != "labels"
            }
            query = {
                key: tensor.to(device).view(batch_size * n_ways * n_queries, seq_len).long()
                for key, tensor in query.items()
                if key != "labels"
            }

            support_features = encoder(**support).embeddings.view(
                batch_size, n_ways * k_shots, seq_len, -1
            )  # [batch_size, n_ways * k_shots, seq_len, d_hidden]
            query_features = encoder(**query).embeddings.view(
                batch_size, n_ways * n_queries, seq_len, -1
            )  # [batch_size, n_ways * n_queries, seq_len, d_hidden]

            if normalize_embeddings:
                support_features = normalize(support_features)
                query_features = normalize(query_features)

            support_features = support_features.cpu().numpy()
            query_features = query_features.cpu().numpy()

            support_targets = support_targets.numpy()
            query_targets = query_targets.numpy()

            support_targets_orig = support_targets_orig.numpy()
            query_targets_orig = query_targets_orig.numpy()

            for batch_idx in range(support_features.shape[0]):
                X_support, y_support, X_query, y_query = prepare_features(
                    support_features[batch_idx],
                    support_targets[batch_idx],
                    support_targets_orig[batch_idx],
                    support_labels[batch_idx],
                    query_features[batch_idx],
                    query_targets[batch_idx],
                    query_targets_orig[batch_idx],
                    query_labels[batch_idx],
                )

                pred_query = classifier(X_support, y_support, X_query)

                for metric, scorer in scorers.items():
                    score = scorer(y_query, pred_query)
                    metric_scores[metric].append(score)

    results: Dict[str, Dict[str, float]] = {}
    for metric, scores in metric_scores.items():
        mean, margin_of_error, _ = mean_confidence_interval(scores, confidence)
        results[metric] = {
            "mean": mean,
            "margin_of_error": margin_of_error,
            "confidence": confidence,
        }

    return results



