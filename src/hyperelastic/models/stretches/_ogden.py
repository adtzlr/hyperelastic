import numpy as np


class Ogden:
    def __init__(self, mu, alpha):
        self.mu = mu
        self.alpha = alpha

    def gradient(self, stretches, statevars):
        dWdλ = 2 * self.mu / self.alpha * stretches ** (self.alpha - 1)
        return dWdλ, statevars

    def hessian(self, stretches, statevars):
        d2Wdλdλ = (
            2 * self.mu / self.alpha * (self.alpha - 1) * stretches ** (self.alpha - 2)
        )
        return np.concatenate([d2Wdλdλ, np.zeros_like(d2Wdλdλ)])
