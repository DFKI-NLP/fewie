import random
from functools import partial

import numpy
import torch
from sklearn import metrics
import logging


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1.0 / 2)
    out = x.div(norm)
    return out


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
        "f1_macro": partial(metrics.f1_score, average="macro"),
    }[metric]


def hinge_contrastive_loss(
    contrastive_embedding: torch.Tensor,
    contrastive_targets: torch.Tensor,
    p: float = 2,
    margin: float = 4,
):
    embedding_left = normalize(torch.squeeze(contrastive_embedding[:, 0, :]))
    embedding_right = normalize(torch.squeeze(contrastive_embedding[:, 1, :]))

    pdist = torch.nn.PairwiseDistance(p=p, eps=1e-06, keepdim=False)
    dist_similar = pdist(embedding_left, embedding_right)
    dist_dissimilar = torch.clamp(margin - dist_similar, min=0)

    loss = (
        1 - contrastive_targets
    ) * dist_similar + contrastive_targets * dist_dissimilar
    return torch.mean(loss)


def batch_where_equal(labels: torch.Tensor, targets_orig: torch.Tensor):
    """Given a batch of contrastive pairs, for each pair, return a position where `labels`
    coincide with `targets_orig`.

    Returns:
        A batch of positions of shape [batch_size, ].
    """
    labels, targets_orig = torch.squeeze(labels), torch.squeeze(targets_orig)
    batch_size = labels.shape[0]
    pos = []
    for text_id in range(batch_size):
        pos_pool = (labels[text_id, :] == targets_orig[text_id]).nonzero().flatten()
        perm = torch.randperm(pos_pool.shape[0])[0]
        pos.append(pos_pool[perm].item())

    pos = torch.Tensor(pos).type(torch.LongTensor)
    return pos
