import torch
from torch import Tensor
import torch.nn.functional as F


def euclidean_distance(x, y):
    return torch.pow(x, 2).sum(dim=1)


def compute_distance_matrix(anchor: Tensor, positive: Tensor, negative: Tensor):
    distance_matrix = torch.zeros(anchor.size(0), 3)
    distance_matrix[:, 0] = euclidean_distance(anchor, anchor)
    distance_matrix[:, 1] = euclidean_distance(anchor, positive)
    distance_matrix[:, 2] = euclidean_distance(anchor, negative)
    return distance_matrix


def batch_all_triplet_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float=0.2):
    distance_matrix: Tensor = compute_distance_matrix(anchor, positive, negative)
    loss = torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 1] + margin)
    loss += torch.max(torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 2] + margin)
    return loss.mean(loss)