import abc
import torch


class FilterState:
    __slots__ = ['x', 'w', 'loglik', 'n']

    def __init__(self, x, w, n, loglik):
        """
        :param x: torch.Tensor[N, D]
        :param w: torch.Tensor[N]
        :param n: int: N
        :param loglik: torch.FloatTensor
        """
        self.w = w
        self.x = x
        self.loglik = loglik
        self.n = n


class ObservationBase(metaclass=abc.ABCMeta):
    # Unsure what a nice structure would be. Fundamentally a namedtuple. Maybe the likelihood method should encode it
    pass


def _normalise(w):
    """
    :param w: torch.Tensor[N]
    :return: torch.Tensor[N]
    """
    return w / w.sum()


class BootstrapFilter:
    def __init__(self,
                 proposal_method,
                 likelihood_method,
                 reweighting_method,
                 min_neff=0.5):
        self.proposal_method = proposal_method
        self.likelihood_method = likelihood_method
        self.reweighting_method = reweighting_method
        self.min_neff = min_neff

    def predict(self, state, _observation):
        """
        :param state: FilterState
        :param _observation: ObservationBase
        :return: FilterState
        """
        return self.proposal_method.apply(state, None)

    def update(self, state, observation):
        """
        :param state: FilterState
        :param observation: ObservationBase
        """
        neff = self._neff(state.w)
        if neff < self.min_neff * state.n:
            state = self._reweight(state, observation)

        likelihood = self.likelihood_method.apply(state, observation)
        prior_weights = state.w
        posterior_weights = likelihood * prior_weights
        previous_log_likelihood = state.loglik
        marginal_likelihood = torch.mean(posterior_weights)
        posterior_weights = _normalise(posterior_weights)
        return FilterState(x=state.x,
                           w=posterior_weights,
                           n=state.n,
                           loglik=previous_log_likelihood + marginal_likelihood.log())

    def _reweight(self, state, _observation):
        x, w = self.reweighting_method.apply(state.x, state.w)
        return FilterState(x=x, w=w, n=state.n, loglik=state.loglik)

    @staticmethod
    def _neff(w):
        """
        :param w: torch.Tensor[N]
        :return: float
        """
        return 1 / (w ** 2).sum()

    def filter(self, state, observation):
        proposal = self.proposal_method(state, observation)
        return self.update(state, observation)