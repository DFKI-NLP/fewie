import torch
from fewie.encoders.encoder import Encoder
from fewie.evaluation.classifiers.classifier import Classifier
from fewie.data.datasets.generic.nway_kshot import ContrastiveNwayKshotDataset
from fewie.evaluation.scenarios.few_shot_linear_readout import (
    mean_confidence_interval,
    prepare_features,
    normalize,
    get_metric,
)
from fewie.evaluation.utils import hinge_contrastive_loss, batch_where_equal

from tqdm import tqdm
from copy import deepcopy
import datasets
from typing import Optional, List, Dict
import logging


def eval_few_show_contrastive_pretraining(
    dataset: datasets.Dataset,
    contrastive_few_shot_dataset: ContrastiveNwayKshotDataset,
    encoder: Encoder,
    classifier: Classifier,
    batch_size: int,
    num_epochs: int,
    weight_decay: float,
    learning_rate: float,
    device: torch.device,
    normalize_embeddings: bool = True,
    confidence: float = 0.95,
    metrics: Optional[List[str]] = None,
):
    if metrics is None:
        metrics = ["accuracy"]
    scorers = {metric: get_metric(metric) for metric in metrics}
    metric_scores: Dict[str, List[float]] = {metric: [] for metric in metrics}

    dataloader = torch.utils.data.DataLoader(
        contrastive_few_shot_dataset, batch_size=batch_size
    )
    n_ways = contrastive_few_shot_dataset.n_ways
    k_shots = contrastive_few_shot_dataset.k_shots
    n_queries = contrastive_few_shot_dataset.n_queries

    for batch in tqdm(dataloader):
        # set model (a pretrained model)
        model = deepcopy(encoder).to(device)
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # load data
        # contrastive: a tuple (contrastive_left, contrastive_right), each of shape:
        #     [batch_size, n_ways * (k_shots + 1), ...]
        # contrastive_targets: [batch_size, n_ways * (k_shots + 1)]
        # contrastive_targets_orig: [batch_size, n_ways * (k_shots + 1), 2]
        # support: [batch_size, n_ways * k_shots, ...]
        # query: [batch_size, n_ways * n_queries, ...]
        # support_targets: [batch_size, n_ways * k_shots]
        # query_targets: [batch_size, n_ways * n_queries]
        (
            contrastive,
            contrastive_targets,
            contrastive_targets_orig,
            support,
            support_targets,
            support_targets_orig,
            query,
            query_targets,
            query_targets_orig,
        ) = batch

        batch_size, _, seq_len = support["input_ids"].shape
        contrastive_left, contrastive_right = contrastive
        contrastive_left = {
            key: tensor.to(device)
            .view(batch_size * n_ways * (k_shots + 1), seq_len)
            .long()
            for key, tensor in contrastive_left.items()
        }
        contrastive_right = {
            key: tensor.to(device)
            .view(batch_size * n_ways * (k_shots + 1), seq_len)
            .long()
            for key, tensor in contrastive_right.items()
        }
        contrastive_targets = contrastive_targets.to(device)
        targets_orig_left = torch.squeeze(contrastive_targets_orig)[:, 0]
        targets_orig_right = torch.squeeze(contrastive_targets_orig)[:, 1]

        # track the positions of our targeted tokens (of specific entity type)
        pos_left = batch_where_equal(contrastive_left["labels"], targets_orig_left)
        pos_right = batch_where_equal(contrastive_right["labels"], targets_orig_right)           
        if pos_left is None or pos_right is None:
            logging.warning("Targeted token out of range of seq_len, skip...")
            continue
        pos_left, pos_right = pos_left.to(device), pos_right.to(device)

        contrastive_left.pop("labels")
        contrastive_right.pop("labels")

        # start training
        for _ in range(num_epochs):
            model.train()
            contrastive_embedding = model(
                contrastive_left, contrastive_right, pos_left, pos_right
            )
            loss = hinge_contrastive_loss(contrastive_embedding, contrastive_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # build prototype
        model.eval()
        with torch.no_grad():
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

            support_features = model.model(**support).last_hidden_state.view(
                batch_size, n_ways * k_shots, seq_len, -1
            )  # [batch_size, n_ways * k_shots, seq_len, d_hidden]
            query_features = model.model(**query).last_hidden_state.view(
                batch_size, n_ways * n_queries, seq_len, -1
            )  # [batch_size, n_ways * n_queries, seq_len, d_hidden]
            del model

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
