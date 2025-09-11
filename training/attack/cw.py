import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional
from torch import nn, optim, Tensor


class MarginLoss(nn.Module):
    """
    Calculates the margin loss max(kappa, (max z_k (x) k != y) - z_y(x)),
    also known as the f6 loss used by the Carlini & Wagner attack.
    """

    def __init__(self, kappa=float('inf'), targeted=False):
        super().__init__()
        self.kappa = kappa
        self.targeted = targeted

    def forward(self, logits, labels):
        correct_logits = torch.gather(logits, 1, labels.view(-1, 1))

        max_2_logits, argmax_2_logits = torch.topk(logits, 2, dim=1)
        top_max, second_max = max_2_logits.chunk(2, dim=1)
        top_argmax, second_argmax = argmax_2_logits.chunk(2, dim=1)
        labels_eq_max = top_argmax.squeeze().eq(labels).float().view(-1, 1)
        labels_ne_max = top_argmax.squeeze().ne(labels).float().view(-1, 1)
        max_incorrect_logits = labels_eq_max * second_max + labels_ne_max * top_max
        max_incorrect_index = labels_eq_max * second_argmax + labels_ne_max * top_argmax
        if self.targeted:
            return (correct_logits - max_incorrect_logits) \
                .clamp(max=self.kappa).squeeze().mean()
        else:
            return (max_incorrect_logits - correct_logits) \
                .clamp(max=self.kappa).squeeze().mean()

def CW_linf(data, label, model, 
        eps = 8/255, epochs=10, 
        is_tg=False, backbone=None, norm='linf', **params):
    loss = MarginLoss(kappa=20)
    if 'device' in params.keys():
        device = params['device']
    else:
        device = 'cpu'

    if model.training:
        model.eval()
        try:
            backbone.eval()
            backbone = backbone.to(device)
        except:
            pass
    alpha = eps / epochs
    model = model.to(device)
    data = data.detach().to(device)
    delta = torch.empty_like(data).uniform_(-eps, eps)
    delta = torch.clamp(data + delta, min=0, max=1).detach() - data
    for _ in range(epochs):
        delta = delta.detach()
        delta.requires_grad = True
        if backbone is not None:
            output = backbone(model(data + delta))
        else:
            output = model(data + delta)
        try:
            cost = loss(output, label.to(device)).to(device)
        except:
            cost = loss(output[0], label.to(device)).to(device)
        if is_tg:
            grad = torch.autograd.grad(cost, delta, 
                                        retain_graph=False, 
                                        create_graph=False)[0]
        else:
            grad = - torch.autograd.grad(cost, delta, 
                                        retain_graph=False, 
                                        create_graph=False)[0]
        
        delta = delta.detach() - alpha * grad.sign()
        delta = torch.clamp(delta, min=-eps, max=eps)
        delta = (torch.clamp(data + delta, min=0, max=1) - data).detach()


    return data + delta