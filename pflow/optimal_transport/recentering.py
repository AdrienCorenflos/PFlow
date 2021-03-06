import numpy as np
import torch

from pflow.base import BaseReweight, NoResampling
from pflow.optimal_transport.transportation_plan import Transport
from pflow.resampling.systematic import SystematicResampling
from pflow.utils.fix_for_geomloss_losses import SamplesLoss


def _learn(x, logw, loss, optim_kwargs, schedule_kwargs, n_steps, init_x, optim_class_name='Adam',
           scheduler_class_name='StepLR'):
    """
    Combine solve_for_state and transport_from_potentials in a "reweighting scheme"
    :param x: torch.Tensor[N, D]
        The input
    :param w: torch.Tensor[N]
        The degenerate log weights
    :param loss: geomloss.SamplesLoss
        Needs to be biased for the potentials to correspond to a proper plan
    :param optim_kwargs: dict
        arguments for optimizer
    :param schedule_kwargs: dict
        arguments for lr scheduler
    :param n_steps: int
        number of steps for optimisation
    :param init_x: tensor
        where to start
    :param optim_class_name: str
        Name of the optimizer
    :param scheduler_class_name: str
        Name of the scheduler
    """

    n = x.shape[0]
    uniform_weights = torch.full_like(logw, np.log(1 / n), requires_grad=False)
    x_i = init_x.clone().detach().requires_grad_(True)

    optimizer_class = getattr(torch.optim, optim_class_name)
    optimizer = optimizer_class([x_i], **optim_kwargs)
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_class_name)

    scheduler = scheduler_class(optimizer, **schedule_kwargs)

    def closure():
        optimizer.zero_grad()
        loss_val = loss(uniform_weights, x_i, logw.detach(), x.detach())
        loss_val.backward()
        return loss_val

    for _ in range(n_steps):
        # loss_val = loss(uniform_weights, x_i, logw.detach(), x.detach())
        optimizer.zero_grad()
        optimizer.step(closure)
        scheduler.step()
    return x_i, uniform_weights


def _incremental_learning(x, w, loss, adam_kwargs, n_steps=5, inner_steps=5):
    """
    Combine solve_for_state and transport_from_potentials in a "reweighting scheme"
    :param x: torch.Tensor[N, D]
        The input
    :param logw: torch.Tensor[N]
        The degenerate logweights
    :param loss: geomloss.SamplesLoss
        Needs to be biased for the potentials to correspond to a proper plan
    :param n_steps: int
        number of steps from degenerate to uniform
    :param inner_steps: int
        inner steps for one set of weights to the next
    :param adam_kwargs: dict
        arguments for adam
    """
    n = x.shape[0]
    ts = np.linspace(0., 1., n_steps + 1)

    x_j = x.clone().detach()
    x_i = x.clone().detach().requires_grad_(True)
    adam = torch.optim.Adam([x_i], **adam_kwargs)

    ones = torch.full_like(w, 1 / n, requires_grad=False)

    for i in range(n_steps):
        w_i = w.detach() * (1 - ts[i]) + ones * ts[i]
        w_i_1 = w.detach() * (1 - ts[i + 1]) + ones * ts[i + 1]
        for _ in range(inner_steps):
            loss_val = loss(w_i_1.log(), x_i, w_i.log(), x_j)
            adam.zero_grad()
            loss_val.backward()
            adam.step()
        x_j.data.copy_(x_i.data)
    return x_j, ones.log()


class LearnBest(BaseReweight):
    def __init__(self,
                 epsilon,
                 geomloss_kwargs,
                 learning_rate,
                 optimizer_kwargs=None,
                 schedule_kwargs=None,
                 n_steps=10,
                 start_from_systematic=False,
                 start_from_regularised=False,
                 jitter=0.,
                 optim_class_name='SGD',
                 scheduler_class_name='StepLR'
                 ):
        """
        Combine solve_for_state and transport_from_potentials in a "reweighting scheme"
        :param epsilon: float
            Blur parameter used in loss
        :param geomloss_kwargs: dict
            dict for Sinkhorn
        :param learning_rate: float
            for the optimizer
        :param optimizer_kwargs: dict
            parameters for the adam optimizer
        :param n_steps: int
            number of epochs
        :param start_from_systematic: bool
            initialisation from the systematic resampling - not proven to be differentiable
        :param start_from_regularised: bool
            initialise from the regularised ensemble transform
        :param jitter: float
            jitter the initial state
        :param optim_class_name: str
            Name of the optimizer
        :param scheduler_class_name: str
            Name of the scheduler
        """
        self.epsilon = epsilon
        geomloss_kwargs.pop('blur', None)
        geomloss_kwargs.pop('potentials', None)
        geomloss_kwargs.pop('debias', None)
        self.learning_rate = learning_rate
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.optim_class_name = optim_class_name
        self.scheduler_class_name = scheduler_class_name
        self.schedule_kwargs = schedule_kwargs
        self.n_steps = n_steps
        self.sample_loss = SamplesLoss(blur=epsilon, is_log=True, debias=True, potentials=False, **geomloss_kwargs)
        if start_from_systematic:
            self._subSample = SystematicResampling()
        elif start_from_regularised:
            self._subSample = Transport(epsilon, **geomloss_kwargs)
        else:
            self._subSample = NoResampling()
        self.jitter = jitter

    def apply(self, x, w, logw):
        """
        Combine solve_for_state and transport_from_potentials in a "reweighting scheme"
        :param x: torch.Tensor[N, D]
            The input
        :param w: torch.Tensor[N]
            The degenerate weights
        """
        init_x, _ = self._subSample.apply(x, w, logw)
        optimizer_kwargs = self.optimizer_kwargs.copy()
        optimizer_kwargs['lr'] = self.learning_rate * init_x.shape[0]
        return _learn(x, logw, self.sample_loss, optimizer_kwargs, self.schedule_kwargs, self.n_steps,
                      init_x + self.jitter * torch.normal(0., 1., init_x.shape),
                      self.optim_class_name, self.scheduler_class_name)


class IncrementalLearning(BaseReweight):
    def __init__(self, epsilon, geomloss_kwargs, adam_kwargs, n_steps=5, inner_steps=5):
        """
        Combine solve_for_state and transport_from_potentials in a "reweighting scheme"
        :param epsilon: float
            Blur parameter used in loss
        :param geomloss_kwargs: dict
            dict for Sinkhorn loss
        :param adam_kwargs: dict
            parameters for the adam optimizer
        :param n_steps: int
            number of epochs
        """
        self.epsilon = epsilon
        self.adam_kwargs = adam_kwargs
        geomloss_kwargs.pop('blur', None)
        geomloss_kwargs.pop('potentials', None)
        geomloss_kwargs.pop('debias', None)
        self.n_steps = n_steps
        self.sample_loss = SamplesLoss(blur=epsilon, is_log=True, debias=True, potentials=False, **geomloss_kwargs)
        self.inner_steps = inner_steps

    def apply(self, x, w, _logw):
        """
        Combine solve_for_state and transport_from_potentials in a "reweighting scheme"
        :param x: torch.Tensor[N, D]
            The input
        :param w: torch.Tensor[N]
            The degenerate weights
        """
        return _incremental_learning(x, w, self.sample_loss, self.adam_kwargs, self.n_steps,
                                     self.inner_steps)

# def recenter_from_proposal(x, w_y, y, lr=1., n_iter=50, **kwargs):
#     uniform = torch.ones_like(w_y)
#     uniform /= uniform.sum()
#     x_new = x.clone().requires_grad_(True)
#     sample_loss = gl.SamplesLoss("sinkhorn", **kwargs)
#     adam = torch.optim.Adam([x_new], lr=lr)
#
#     for _ in range(n_iter):
#         adam.zero_grad()
#         loss = sample_loss(uniform, x_new, w_y, y)
#         loss.backward()
#         adam.step()
#     return x_new.clone()
#
#
# def recenter_from_target(w_y, y, lr=0.5, n_ts=5, n_iter=5, **kwargs):
#     ts = torch.linspace(1 / n_ts, 1, n_ts, requires_grad=False)
#     sample_loss = gl.SamplesLoss("sinkhorn", **kwargs)
#     uniform = torch.ones_like(w_y, requires_grad=False)
#     uniform /= uniform.sum()
#     y_1 = y.clone()
#     w_0 = w_y
#     for t in ts:
#         w_1 = (w_y * (-t + 1.) + t * uniform)
#         y_0 = y_1.clone()
#         y_0_clone = y_0.clone()
#         y_1 = y_0.detach().requires_grad_(True)
#         adam = torch.optim.Adam([y_1], lr=lr)
#         for _ in range(n_iter):
#             adam.zero_grad()
#             loss = sample_loss(w_1, y_1, w_0, y_0_clone)
#             loss.backward()
#             adam.step()
#         w_0 = w_1.detach()
#     return y_1.clone()
#
#
# def main():
#     import time
#     import matplotlib.pyplot as plt
#     torch.random.manual_seed(0)
#     n = 300
#     x = torch.randn(n, 1)
#     y, idx = torch.randn(n, 1).sort(0)
#     w_y = 0.5 + torch.rand(n) * 0.5
#     w_y /= w_y.sum()
#     w_y[:100] = 0.
#     print((w_y - 1 / n).abs().mean())
#     print(y[100])
#     tic = time.time()
#     from_proposal = recenter_from_proposal(x, w_y, y, backend='tensorized').detach().numpy()
#     print(time.time() - tic)
#
#     tic = time.time()
#     from_target = recenter_from_target(w_y, y, n_ts=3, n_iter=10, lr=0.25, backend='tensorized').detach().numpy()
#     print(time.time() - tic)
#
#     plt.hist(from_proposal.squeeze(), bins=30, alpha=0.5, label='from_proposal', density=True)
#     plt.hist(from_target.squeeze(), bins=30, alpha=0.5, label='from_target', density=True)
#     plt.hist(y.detach().squeeze().numpy().tolist(), weights=w_y.detach().numpy(), bins=30, alpha=0.5, label='initial',
#              density=True)
#     plt.legend()
#     plt.show()
#
#
# if __name__ == '__main__':
#     main()
