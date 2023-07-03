import numpy as np


class Ogden:
    def __init__(self, mu, alpha):
        self.mu = mu
        self.alpha = alpha

    def gradient(self, stretches, statevars):
        dWdλα = np.zeros_like(stretches)

        for mu, alpha in zip(self.mu, self.alpha):
            dWdλα += 2 * mu / alpha * stretches ** (alpha - 1)

        return dWdλα, statevars

    def hessian(self, stretches, statevars):
        d2Wdλαdλα = np.zeros_like(stretches)

        for mu, alpha in zip(self.mu, self.alpha):
            d2Wdλαdλα += 2 * mu / alpha * (alpha - 1) * stretches ** (alpha - 2)

        return [*d2Wdλαdλα, None, None, None]
