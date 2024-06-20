"""
@Time    : 2024/3/11 08:20
@Author  : YuZhang
@File    : losses.py
"""
import torch
import torch.nn.functional as F
import numpy as np
def get_align_loss(embedding1, embedding2, alpha=2):
    embedding1 = torch.nn.functional.normalize(embedding1, dim=-1)
    embedding2 = torch.nn.functional.normalize(embedding2, dim=-1)

    return torch.mean((embedding1 - embedding2).norm(p=2, dim=1).pow(alpha))
def get_uniform_loss(embedding, t=2):
    embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return torch.pdist(embedding, p=2).pow(2).mul(-t).exp().mean().log()

def InfoNCE(view1, view2, temperature):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()
