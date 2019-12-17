from pflow.base import BaseReweight
import torch
import numpy as np
import math


class SystematicResampling(BaseReweight):
    def apply(self, x, w, _logw):
        u = torch.rand(())
        n = x.shape[0]
        probs = (torch.arange(n, dtype=w.dtype) + u) / n
        cumsum = w.cumsum(0)
        indices = np.searchsorted(cumsum.detach(), probs)
        return x[indices, :], torch.full_like(w, math.log(1 / n))
