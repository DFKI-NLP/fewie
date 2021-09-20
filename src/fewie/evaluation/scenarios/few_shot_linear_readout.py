from typing import Dict, List, Optional, Tuple

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
    """Computes the mean and error margin of given data for a given confidence level.

    Args:
        data: A list of data (in this case F1-scores).
        confidence: The coverage probability we want to achieve with error margin.

    Returns:
        Mean and margin error of data, where mean equals the arithmetic average of data,
        and margin error means: `[mean - margin_error, mean + margin_error]` covers
        `confidence`*100% of the data points from `data`.
    """
    array = np.array(data)
    num = len(array)
    ddof = num - 1
    mean, std_error_mean = np.mean(array), scipy.stats.sem(array)
    margin_of_error = std_error_mean * scipy.stats.t._ppf(
        (1.0 + confidence) / 2.0, ddof
    )
    return (
        mean,
        margin_of_error,
        scipy.stats.t.interval(
            0.95, ddof, loc=np.mean(array), scale=scipy.stats.sem(array)
        ),
    )


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalizes a vector with its L2-norm.

    Args:
        x: The vector to be normalized.

    Returns:
        The normalized vector of the same shape.
    """
    norm = x.pow(2).sum(1, keepdim=True).pow(1.0 / 2)
    out = x.div(norm)
    return out


def prepare_features(
    support_features: np.ndarray,
    support_targets: np.ndarray,
    support_labels: np.ndarray,
    query_features: np.ndarray,
    query_targets: np.ndarray,
    query_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepares the features (token-level) from the support/query set (sentence-level).

    Since the support and query sets are sampled on a sentence-level (which means for each
    to-be-classified token, we just take its whole sentence for contextual embedding), and 
    NER task is on token-level, in this function, we aim to take only the corresponding 
    position from the `seq_len`-length embedding for each sample.

    Args:
        support_features: The embedding of the whole sentence containing certain entitie for \
            the support set, of shape: `[batch_size * n_ways * k_shots, seq_len, d_hidden]`.\n
        support_targets: The original class-ids (therefore might not be continuous) \
            for the support set, of shape: `[batch_size * n_ways * k_shots]`.\n
        support_labels: The labels of all the tokens in a sentence for the support set, \
            of shape: `[batch_size * n_ways * k_shots, seq_len, ]`.\n
        query_features: The embeddings of the sentences containing entities for the query set, \
            of shape: `[batch_size * n_ways * n_queries, seq_len, d_hidden]`.\n
        query_targets: The (original) class-ids for the query set, of shape: \
            `[batch_size * n_ways * k_shots]`.\n
    
    Returns:
        X_support: The contextual embedding of only the wanted tokens for the support set, \
            of shape: `[batch_size * n_wanted_tokens, h_didden]`.\n
        y_support: The encoded (because multi-classification is applied later) class-ids for the \
            support set, of shape `[batch_size * n_wanted_tokens, ]`.\n
        X_query: The contextual embedding of only the wanted tokens for the query set, \
            of shape: `[batch_size * n_wanted_tokens, h_hidden]`.\n
        y_support: The encoded class-ids for the query set, of shape \
            `[batch_size * n_wanted_tokens, ]`.
    """
    X_support = []
    y_support = []
    for i, (target, labels) in enumerate(zip(support_targets, support_labels)):
        # take only the position with the wanted tokens
        mask = labels == target
        features = support_features[i, mask, :]
        X_support.append(features)
        y_support.extend([target] * features.shape[0])

    X_query = []
    y_query = []
    for i, (target, labels) in enumerate(zip(query_targets, query_labels)):
        mask = labels == target
        features = query_features[i, mask, :]
        X_query.append(features)
        y_query.extend([target] * features.shape[0])

    X_support = np.concatenate(X_support, axis=0)
    y_support = np.array(y_support)
    X_query = np.concatenate(X_query, axis=0)
    y_query = np.array(y_query)
    return (X_support, y_support, X_query, y_query)


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
    """Performs evaluation using prototypes of contextual embeddings and linear-readout method
    as classifier top.
    """
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
        # Each "batch" corresponds to an independent experiment run.
        for batch in tqdm(dataloader):
            # support: [batch_size, n_ways * k_shots, ...]
            #   with columns: `attention_mask`, `input_ids`, `labels`, `token_type_ids`
            # query: [batch_size, n_ways * n_queries, ...]
            # support_targets: [batch_size, n_ways * k_shots]
            # query_targets: [batch_size, n_ways * n_queries]
            (
                support,
                support_targets,
                query,
                query_targets,
            ) = batch

            batch_size, _, seq_len = support["input_ids"].shape

            support_labels = support["labels"].cpu().numpy()
            query_labels = query["labels"].cpu().numpy()

            support = {
                key: tensor.to(device)
                .view(batch_size * n_ways * k_shots, seq_len)
                .long()
                for key, tensor in support.items()
                if key != "labels"
            }
            query = {
                key: tensor.to(device)
                .view(batch_size * n_ways * n_queries, seq_len)
                .long()
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

            for batch_idx in range(support_features.shape[0]):
                X_support, y_support, X_query, y_query = prepare_features(
                    support_features[batch_idx],
                    support_targets[batch_idx],
                    support_labels[batch_idx],
                    query_features[batch_idx],
                    query_targets[batch_idx],
                    query_labels[batch_idx],
                )

                pred_query = classifier(X_support, y_support, X_query)

                # prepare the entity-label list
                entity_label_list = set(y_query)
                entity_label_list.discard(0)

                for metric, scorer in scorers.items():
                    score = scorer(y_query, pred_query, labels=list(entity_label_list))
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
