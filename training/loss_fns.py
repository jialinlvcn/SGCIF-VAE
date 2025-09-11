from utils import trades, advt
import torch.nn as nn
import torch
from utils.hsic import hsic_normalized

class AdersialTrainingLoss(nn.Module):
    def __init__(self, method='TRADES'):
        super(AdersialTrainingLoss, self).__init__()
        self.method = method

    def forward(self, model, x, x_adv, y):
        if self.method.lower() == 'trades':
            return trades(model, x, x_adv, y)
        elif self.method.lower() == 'at':
            return advt(model, x_adv, y)
        else:
            raise ValueError(f"Unknown adversarial training method: {self.method.lower()} ")

class CIFLoss(nn.Module):
    def __init__(self, alpha=0.1, lam=0.1, beta=0.1, at_method='at', dim=2048):
        super(CIFLoss, self).__init__()
        self.alpha = alpha
        self.lam = lam
        self.beta = beta
        self.dim = dim
        self.at_loss_fns = AdersialTrainingLoss(method=at_method)

    def forward(self, model, x, x_adv, gt):
        # Compute adversarial training loss
        ce_loss, out = self.at_loss_fns(model, x, x_adv, gt)
        out, hi, xi, mu, logvar = out

        # Compute KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= x.size(0) * 3 * self.dim

        # Compute Information Bottleneck loss using HSIC
        hsic_hy_xt = hsic_normalized(
                hi, x_adv.flatten(start_dim=1), sigma=5, use_cuda=True
        )
        hsic_hy_ot = hsic_normalized(
                hi, xi.flatten(start_dim=1), sigma=5, use_cuda=True
        )
        ib_loss = hsic_hy_xt - self.lam * hsic_hy_ot

        # Total loss
        totel_loss = ce_loss + self.alpha * ib_loss + self.beta * kl_loss

        return totel_loss, out
