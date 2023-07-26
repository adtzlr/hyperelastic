import numpy as np


class Ogden:
    r"""Ogden isotropic hyperelastic material formulation based on the principal
    stretches. The strain energy density per unit undeformed volume is given as a sum of
    individual contributions from the principal stretches.

    ..  math::

        \psi(\lambda_\alpha) &= \sum_\alpha \psi_\alpha(\lambda_\alpha)

        \psi_\alpha(\lambda_\alpha) &= \frac{2 \mu}{k^2} \left(
            \lambda_\alpha^k - 1 \right)

    The first partial derivatives of the strain energy density w.r.t. the
    principal stretches are required for the principal values of the stress.

    ..  math::

        \frac{\partial \psi}{\partial \lambda_\alpha} = \sum_\alpha
            \frac{2 \mu}{k} \lambda_\alpha^{k - 1}

    Furthermore, the second partial derivatives of the strain energy density w.r.t. the
    principal stretches, necessary for the principal components of the elastic tangent
    moduli, are carried out.

    ..  math::

        \frac{\partial^2 \psi}{\partial \lambda_\alpha~\partial \lambda_\alpha} =
            \sum_\alpha \frac{2 \mu (k-1)}{k} \lambda_\alpha^{k - 2}

    """

    def __init__(self, mu, alpha):
        """Initialize the Third Order Deformation material formulation with its
        parameters.
        """

        self.mu = mu
        self.alpha = alpha

    def gradient(self, stretches, statevars):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the principal stretches."""

        dWdλα = np.zeros_like(stretches)

        for mu, alpha in zip(self.mu, self.alpha):
            dWdλα += 2 * mu / alpha * stretches ** (alpha - 1)

        return dWdλα, statevars

    def hessian(self, stretches, statevars):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the principal stretches."""

        d2Wdλαdλα = np.zeros_like(stretches)

        for mu, alpha in zip(self.mu, self.alpha):
            d2Wdλαdλα += 2 * mu / alpha * (alpha - 1) * stretches ** (alpha - 2)

        return [*d2Wdλαdλα, None, None, None]
