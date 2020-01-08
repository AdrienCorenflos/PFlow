import abc


class BaseReweight(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, x, w, logw):
        """
         Combine solve_for_state and transport_from_potentials in a "reweighting scheme"
         :param x: torch.Tensor[N, D]
             The input
         :param w: torch.Tensor[N]
             The degenerate weights
         :param logw: torch.Tensor[N]
            log weights
         :return
            x, logw the corrected particles and weights
         """


class NoResampling(BaseReweight):
    def apply(self, x, w, logw):
        return x, logw