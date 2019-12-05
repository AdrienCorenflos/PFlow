import geomloss
from geomloss.utils import squared_distances
import torch

from pflow.utils.fix_for_geomloss import sinkhorn_loop

geomloss.sinkhorn_samples.sinkhorn_loop = sinkhorn_loop
# This is to fix the missing gradient for weights


def transport_from_potentials(x, f, g, eps, w, n):
    """
    To get the transported particles from the sinkhorn iterates

    :param x: torch.Tensor[N, D]
        Input: the state variable
    :param f: torch.Tensor[N]
        Potential, output of the sinkhorn iterates
    :param g: torch.Tensor[N]
        Potential, output of the sinkhorn iterates
    :param eps: float
    :param w: torch.Tensor[N]
    :param n: int
    :return: torch.Tensor[N, D], torch.Tensor[N]
    """
    cost = squared_distances(x, x) / 2.
    fg = f.T + g
    transport = torch.exp((fg - cost)/eps**2) * w.unsqueeze(1)
    return transport.T @ x, torch.full_like(f, 1 / n).squeeze()


def solve_for_state(x, w, loss, n):
    """
    Use Geomloss to find the transportation potentials for (x, w) to (w, 1/N)

    :param x: torch.Tensor[N, D]
        The input
    :param w: torch.Tensor[N]
        The degenerate weights
    :param loss: geomloss.SamplesLoss
        Needs to be biased for the potentials to correspond to a proper plan
    :param n: int
    :return: torch.Tensor[N], torch.Tensor[N]
        the potentials
    """
    uniform_weights = torch.full_like(w, 1/n)
    alpha, beta = loss(uniform_weights, x, w, x)
    return alpha, beta


def reweight(x, w, loss, eps):
    """
    Combine solve_for_state and transport_from_potentials in a "reweighting scheme"
    :param x: torch.Tensor[N, D]
        The input
    :param w: torch.Tensor[N]
        The degenerate weights
    :param loss: geomloss.SamplesLoss
        Needs to be biased for the potentials to correspond to a proper plan
    :param eps: float
        Blur parameter used in loss
        """
    n = x.shape[0]
    alpha, beta = solve_for_state(x, w, loss, n)
    x_tilde, w_tilde = transport_from_potentials(x, alpha, beta, eps, w, n)
    return x_tilde, w_tilde


class Reweighting:
    def __init__(self, epsilon, **geomloss_kwargs):
        self.epsilon = epsilon
        geomloss_kwargs.pop('blur', None)
        geomloss_kwargs.pop('potentials', None)
        geomloss_kwargs.pop('debias', None)
        self.sample_loss = geomloss.SamplesLoss(blur=epsilon, debias=False, potentials=True, **geomloss_kwargs)

    def apply(self, x, w):
        return reweight(x, w, self.sample_loss, self.epsilon)



