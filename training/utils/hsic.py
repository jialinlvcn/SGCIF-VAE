# code from https://github.com/choasma/HSIC-bottleneck

import torch
import numpy as np


class DivisionByZeroError(Exception):
    pass


def distmat(X):
    """distance matrix"""
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def sigma_estimation(X, Y):
    """sigma from median distance"""
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1e-2:
        med = 1e-2
    return med


def kernelmat(X, sigma, k_type="gaussian"):
    """kernel matrix baker"""
    device = X.device
    m = int(X.size()[0])
    H = (torch.eye(m) - (1.0 / m) * torch.ones([m, m])).to(device)

    if k_type == "gaussian":
        Dxx = distmat(X)

        if sigma:
            variance = 2.0 * sigma * sigma * X.size()[1]
            Kx = (
                torch.exp(-Dxx / variance).type(torch.FloatTensor).to(device)
            )  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X, X)
                if ((2.0 * sx * sx) < 0).any():
                    raise DivisionByZeroError("除数不能为零")
                Kx = (
                    torch.exp(-Dxx / (2.0 * sx * sx)).type(torch.FloatTensor).to(device)
                )
            except RuntimeError:
                raise RuntimeError(
                    "Unstable sigma {} with maximum/minimum input ({},{})".format(
                        sx, torch.max(X), torch.min(X)
                    )
                )

    ## Adding linear kernel
    elif k_type == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor).to(device)

    Kxc = torch.mm(Kx, H)

    return Kxc


def hsic_regular(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """ """
    device = x.device
    Kxc = kernelmat(x, sigma).to(device)
    Kyc = kernelmat(y, sigma).to(device)
    KtK = torch.mul(Kxc, Kyc.t()).to(device)
    Pxy = torch.mean(KtK).to(device)
    return Pxy


def hsic_normalized(x, y, sigma=None, use_cuda=True, to_numpy=False):
    """ """
    device = x.device
    Pxy = hsic_regular(x, y, sigma).to(device)
    if (hsic_regular(x, x, sigma) < 0).any():
        raise DivisionByZeroError("负数不能开根号")
    if (hsic_regular(y, y, sigma) < 0).any():
        raise DivisionByZeroError("负数不能开根号")
    Px = torch.sqrt(hsic_regular(x, x, sigma).to(device))
    Py = torch.sqrt(hsic_regular(y, y, sigma).to(device))
    if (Px * Py == 0).any():
        raise DivisionByZeroError("除数不能为0")
    thehsic = Pxy / (Px * Py)
    return thehsic
