import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, device
from torch.nn import Module

from math import sqrt, log, cos, sin, pi


class ArcFace(Module):
    def __init__(self, in_features: int, out_features: int, device:device, scale=None, margin=0.3):
        super(ArcFace, self).__init__()
        self.device = device
        if scale == None:
            self.scale = sqrt(2) * log(out_features - 1)
        else:
            self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_normal_(self.weights)
        self.cos_m = cos(margin)
        self.sin_m = sin(margin)
        self.th = cos(pi - margin)
        self.mm = sin(pi - margin) * margin

    def forward(self, x: Tensor, labels) -> Tensor:
        cosine = F.linear(x, F.normalize(self.weights))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output
