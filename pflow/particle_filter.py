import abc
import math


def _log_mean_exp(logw):
    max_ = logw.max()
    v = (logw - max_).exp()
    return max_ + v.sum().log()


class LikelihoodMethodBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, state, observation, log=True):
        """
        :param state: FilterState
            current state of the filter
        :param observation: ObservationBase
        :param log: bool
            Return the loglikelihood of the state or the likelihood
        :return: torch.float
            The log-likelihood or the likelihood
        """


class ProposalMethodBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, state, observation):
        """
        :param state: FilterState
            current state of the filter
        :param observation: ObservationBase
        :return: State
            Proposal state
        """


class FilterState:
    __slots__ = ['x', 'w', 'logw', 'loglik', 'n', 'marginal_log_likelihood']

    def __init__(self, x, logw, n, loglik, marginal_log_likelihood=0.):
        """
        :param x: torch.Tensor[N, D]
        :param logw: torch.Tensor[N]
        :param n: int: N
        :param loglik: torch.FloatTensor
        :param marginal_log_likelihood: torch.FloatTensor
        """
        self.w = logw.exp()
        self.logw = logw
        self.x = x
        self.loglik = loglik
        self.n = n
        self.marginal_log_likelihood = marginal_log_likelihood


class ObservationBase(metaclass=abc.ABCMeta):
    # Unsure what a nice structure would be. Fundamentally a namedtuple. Maybe the likelihood method should encode it
    pass


def _normalise(w, log=True):
    """
    :param w: torch.Tensor[N]
    :param log: bool
    :return: torch.Tensor[N]
    """
    if log:
        return w - w.logsumexp(0)
    return w / w.sum()


class BootstrapFilter:
    MIN_W = 1 / 100

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

        likelihood = self.likelihood_method.apply(state, observation, log=True)
        prior_log_weights = state.logw
        posterior_log_weights = likelihood + prior_log_weights
        posterior_log_weights = posterior_log_weights.clamp(-10*math.log(state.n), 0.)
        previous_log_likelihood = state.loglik
        marginal_log_likelihood = _log_mean_exp(posterior_log_weights)
        posterior_weights = _normalise(posterior_log_weights, log=True)
        log_lik_increment = marginal_log_likelihood - state.marginal_log_likelihood
        return FilterState(x=state.x,
                           logw=posterior_weights,
                           n=state.n,
                           loglik=previous_log_likelihood + log_lik_increment,
                           marginal_log_likelihood=marginal_log_likelihood)

    def _reweight(self, state, _observation):
        x, logw = self.reweighting_method.apply(state.x, state.w, state.logw)
        return FilterState(x=x,
                           logw=logw,
                           n=state.n,
                           loglik=state.loglik,
                           marginal_log_likelihood=0.)

    @staticmethod
    def _neff(w):
        """
        :param w: torch.Tensor[N]
        :return: float
        """
        return 1 / (w ** 2).sum()

    def filter(self, state, observation):
        proposal = self.proposal_method(state, observation)
        return self.update(proposal, observation)
