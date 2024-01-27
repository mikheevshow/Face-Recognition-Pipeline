import torch
from torch import Tensor


def euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute Eucledean distance between two  tensors
    :param x: the first tensor
    :param y: the second tensor
    :return: the Euclidean distance between two tensors
    """
    return torch.pow(x, 2).sum(dim=1)


def compute_distance_matrix(anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
    """
    Compute distance matrix between two anchor, positive and negative tensors
    :param anchor: anchor tensor
    :param positive: positive tensor
    :param negative: negative tensor
    :return: distance matrix
    """
    distance_matrix = torch.zeros(anchor.size(0), 3)
    distance_matrix[:, 0] = euclidean_distance(anchor, anchor)
    distance_matrix[:, 1] = euclidean_distance(anchor, positive)
    distance_matrix[:, 2] = euclidean_distance(anchor, negative)
    return distance_matrix


def batch_all_triplet_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 0.2):
    distance_matrix: Tensor = compute_distance_matrix(anchor, positive, negative)
    loss = torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 1] + margin)
    loss += torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 2] + margin)
    return loss.mean(loss)


def batch_hard_triplet_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin=0.2) -> Tensor:
    distance_matrix: Tensor = compute_distance_matrix(anchor, positive, negative)
    hard_negative = torch.argmax(distance_matrix[:, 2])
    loss = torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 1] + margin)
    loss += torch.max(torch.tensor(0.0), distance_matrix[:, 0][hard_negative] - distance_matrix[:, 2] + margin)
    return torch.mean(loss)


def triplet_loss(anchor: Tensor,
                 positive: Tensor,
                 negative: Tensor,
                 margin: float = 0.2,
                 strategy: str = 'all') -> Tensor:
    if strategy == 'all':
        return batch_all_triplet_loss(anchor, positive, negative, margin)
    elif strategy == 'hard':
        return batch_hard_triplet_loss(anchor, positive, negative, margin)
    else:
        raise ValueError(f'Unknown strategy {strategy}')

