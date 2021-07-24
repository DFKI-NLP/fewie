import random
from functools import partial

import numpy
import torch
from sklearn import metrics


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
    margin: float = 2,
) -> torch.Tensor:
    """Compute the Hinge (Simple) Contrastive Loss over a batch.

    Args: 
        contrastive_embedding: The contextual token-level embedding given by a contrastive \
            encoder, of size `[..., 2, embedding_size]`.
        contrastive_targets: The classes where the two contrastive tokens come from, of size \
            `[batch_size, n_ways * (k_shots + 1)]`.
        p: Specify which p-norm we want.
        margin: The maximum edge of a dissimilar pair.
    
    Returns: 
        A scalar representing the batch-mean of the defined loss.
    """
    embedding_left = normalize(torch.squeeze(contrastive_embedding[:, 0, :]))
    embedding_right = normalize(torch.squeeze(contrastive_embedding[:, 1, :]))

    pdist = torch.nn.PairwiseDistance(p=p, eps=1e-06, keepdim=False)
    dist_similar = pdist(embedding_left, embedding_right)
    dist_dissimilar = torch.clamp(margin - dist_similar, min=0)

    loss = (
        1 - contrastive_targets
    ) * dist_similar + contrastive_targets * dist_dissimilar
    return torch.mean(loss)


def n_pair_loss(
    contrastive_embedding: torch.Tensor, contrastive_targets_orig: torch.Tensor
) -> torch.Tensor:
    """Compute the N-pair Loss over a batch.

    Args: 
        contrastive_embedding: The contextual token-level embedding given by a contrastive \
            encoder, of size `[..., 2, embedding_size]`.
        contrastive_targets_orig: The classes where the two contrastive tokens come from, of size \
            `[batch_size, n_ways * (k_shots + 1), 2]`.

    Returns: 
        Mean of the defined loss given by a scalar.
    """
    # both of shape `[n_ways * (k_shots + 1), embedding_size]` after squeezing
    embedding_left = normalize(torch.squeeze(contrastive_embedding[:, 0, :]))
    embedding_right = normalize(torch.squeeze(contrastive_embedding[:, 1, :]))

    # [batch_size * n_ways * (k_shots + 1), 2]
    contrastive_targets_orig = torch.squeeze(contrastive_targets_orig)

    centered_classes = torch.unique(contrastive_targets_orig[:, 0])
    n_pair_losses = 0
    for cls in centered_classes:
        # compute denominator from all pairs
        denominator = 0
        pair_ids = (contrastive_targets_orig[:, 0] == cls).nonzero().flatten()
        for id in pair_ids:
            denominator += torch.dot(embedding_left[id], embedding_right[id])

        # compute numerator from similar pair
        id = (contrastive_targets_orig[pair_ids, 1] == cls).nonzero().flatten()
        id = int(id.item())
        numerator = torch.dot(embedding_left[id], embedding_right[id])

        # compute n-pair loss centering one class with 1 positive and N negative contrastives
        loss = -torch.log(numerator / denominator)
        n_pair_losses += loss

    return n_pair_losses / len(centered_classes)


def batch_where_equal(labels: torch.Tensor, targets_orig: torch.Tensor) -> torch.Tensor:
    """Given a batch of contrastive pairs, for each pair, return a position where `labels`
    coincide with `targets_orig`, which is used to track the wanted token in a sentence.

    Args:
        labels: The labels for tokens in each text of a batch, of shape \
            `[batch_size * n_ways * (k_shots + 1), seq_len]`.
        targets_orig: The target entity class, of shape `[batch_size, n_ways * (k_shots + 1)]`.

    Returns:
        A batch of positions of shape `[batch_size, n_ways * (k_shots + 1)]`.
    """
    labels, targets_orig = torch.squeeze(labels), torch.squeeze(targets_orig)
    pos = []
    for text_id in range(labels.shape[0]):
        pos_pool = (labels[text_id, :] == targets_orig[text_id]).nonzero().flatten()
        if pos_pool.nonzero().flatten().shape[0] == 0:
            return None
        perm = torch.randperm(pos_pool.shape[0])[0]
        pos.append(pos_pool[perm].item())

    pos = torch.Tensor(pos).type(torch.LongTensor)
    return pos
