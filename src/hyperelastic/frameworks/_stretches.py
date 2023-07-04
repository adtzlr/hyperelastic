import numpy as np

from ..math import cdya, dya, eigh, transpose


class Stretches:
    r"""The Framework for a Total-Lagrangian stretch-based isotropic hyperelastic
    material formulation provides the material behaviour-independent parts for
    evaluating the second Piola-Kirchhoff stress tensor as well as its associated
    fourth-order elasticity tensor.

    The gradient as well as the hessian of the strain energy function are carried out
    w.r.t. the right Cauchy-Green deformation tensor. Hence, the work-conjugate stress
    tensor is one half of the second Piola-Kirchhoff stress tensor and the fourth-order
    elasticitiy tensor used here is a quarter of the Total-Lagrangian elasticity tensor.

    ..  math::

        \psi(\boldsymbol{C}) = \psi(\lambda_\alpha(\boldsymbol{C}))

    The principal stretches (the square roots of the eigenvalues) of the left or right
    Cauchy-Green deformation tensor are obtained by the solution of the eigenvalue
    problem,

    ..  math::

        \left( \boldsymbol{C} - \lambda^2_\alpha \boldsymbol{I} \right)
            \boldsymbol{N}_\alpha = \boldsymbol{0}

    where the Cauchy-Green deformation tensors eliminate the rigid body rotations
    of the deformation gradient and serve as a quadratic change-of-length measure of
    the deformation.

    ..  math::

        \boldsymbol{C} &= \boldsymbol{F}^T \boldsymbol{F}

        \boldsymbol{b} &= \boldsymbol{F} \boldsymbol{F}^T

    The first partial derivative of the strain energy function w.r.t. a principal
    stretch

    ..  math::

        \psi_{,\alpha} = \frac{\partial \psi}{\partial \lambda_\alpha}

    and the partial derivative of a principal strech w.r.t. the right Cauchy-Green
    deformation tensor is defined

    ..  math::

        \frac{\partial \lambda_\alpha}{\partial \boldsymbol{C}} =
            \frac{\partial (\lambda^2_\alpha)^{1/2}}{\partial \boldsymbol{C}} =
            \frac{1}{2 \lambda_\alpha} \boldsymbol{M}_\alpha

    with the eigenbase as the dyadic (outer vector) product of eigenvectors.

    ..  math::

        \boldsymbol{M}_\alpha = \boldsymbol{N}_\alpha \otimes \boldsymbol{N}_\alpha

    The second Piola-Kirchhoff stress tensor is formulated by the application of the
    chain rule and a sum of all principal stretch contributions.

    ..  math::

        \frac{\partial \psi}{\partial \boldsymbol{C}} &=
            \sum_\alpha \frac{\partial \psi}{\partial \lambda_\alpha}
                \frac{\partial \lambda_\alpha}{\partial \boldsymbol{C}}

        \boldsymbol{S} &= 2 \frac{\partial \psi}{\partial \boldsymbol{C}}

    Furthermore, the second partial derivatives of the elasticity tensor are carried
    out.

    ..  math::

        \frac{\partial^2 \psi}{\partial \boldsymbol{C}~\partial \boldsymbol{C}} &=
            \sum_\alpha \sum_\beta \frac{\partial^2 \psi}
            {\partial \lambda_\alpha~\partial \lambda_\beta}
            \frac{\partial \lambda_\alpha}{\partial \boldsymbol{C}} \otimes
            \frac{\partial \lambda_\beta}{\partial \boldsymbol{C}}

            &+ \sum_\alpha \sum_{\beta \ne \alpha} \frac{
                \frac{\partial \psi}{\partial \lambda^2_\alpha} -
                \frac{\partial \psi}{\partial \lambda^2_\beta}
            }{\lambda^2_\alpha - \lambda^2_\beta} \left( \boldsymbol{M}_\alpha \odot
                \boldsymbol{M}_\beta + \boldsymbol{M}_\beta \odot
            \boldsymbol{M}_\alpha \right)


    ..  math::

        \mathbb{C} = 4 \frac{\partial^2 \psi}
            {\partial \boldsymbol{C}~\partial \boldsymbol{C}}

    In case of repeated equal principal stretches, the rule of d'Hospital is applied.

    ..  math::

        \lim_{\lambda^2_\beta \rightarrow \lambda^2_\alpha} \left(
                \frac{\frac{\partial \psi}{\partial \lambda^2_\alpha} -
                    \frac{\partial \psi}{\partial \lambda^2_\beta}
                }{\lambda^2_\alpha - \lambda^2_\beta}
            \right) = \left(
            - \frac{\partial^2 \psi}{\partial \lambda^2_\alpha~\partial \lambda^2_\beta}
            + \frac{\partial^2 \psi}{\partial \lambda^2_\beta~\partial \lambda^2_\beta}
        \right)

    """

    def __init__(self, material, nstatevars=0, parallel=False):
        """Initialize the Framework for an invariant-based isotropic hyperelastic
        material formulation."""

        self.parallel = parallel
        self.material = material

        self.x = [np.eye(3), np.zeros(nstatevars)]

    def gradient(self, C, statevars):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the right Cauchy-Green deformation tensor (one half of the second Piola
        Kirchhoff stress tensor)."""

        self.λ, self.M = eigh(C, fun=np.sqrt)

        self.dWdλ, statevars_new = self.material.gradient(self.λ, statevars)
        self.dWdλ = np.asarray(self.dWdλ)

        self.dWdλC = self.dWdλ / (2 * self.λ)

        dWdC = np.sum(np.expand_dims(self.dWdλC, axis=1) * self.M, axis=0)

        return dWdC, statevars_new

    def hessian(self, C, statevars):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the right Cauchy-Green deformation tensor (a quarter of the Lagrangian
        fourth-order elasticity tensor associated to the second Piola-Kirchhoff stress
        tensor)."""

        dWdC, statevars = self.gradient(C, statevars)
        d2Wdλdλ = self.material.hessian(self.λ, statevars)

        if not isinstance(d2Wdλdλ, np.ndarray):
            newshape = self.λ[0].shape
            _d2Wdλdλ = np.zeros((6, *self.λ[0].shape))
            for a in range(6):
                if d2Wdλdλ[a] is not None:
                    _d2Wdλdλ[a] = np.broadcast_to(d2Wdλdλ[a], newshape)
            d2Wdλdλ = _d2Wdλdλ

        λ = self.λ
        M = self.M
        dWdλC = self.dWdλC

        a = [0, 1, 2, 0, 1, 0]
        b = [0, 1, 2, 1, 2, 2]

        d2WdλC2 = d2Wdλdλ / (4 * λ[a] * λ[b])
        d2WdλC2[:3] -= dWdλC / (2 * λ**2)

        d2WdCdC = np.zeros((6, 6, *dWdλC.shape[1:]))

        a = [0, 1, 2, 0, 1, 0]
        b = [0, 1, 2, 1, 2, 2]

        for m, (α, β) in enumerate(zip(a, b)):
            M4 = dya(M[α], M[β])
            d2WdCdC += d2WdλC2[m] * M4

            if β != α:
                v = λ[α] ** 2 - λ[β] ** 2
                mask = np.isclose(v, 0)

                f = np.zeros_like(v)
                f[~mask] = (dWdλC[α][~mask] - dWdλC[β][~mask]) / v[~mask]
                f[mask] = d2WdλC2[β][mask] - d2WdλC2[m][mask]

                d2WdCdC += d2WdλC2[m] * transpose(M4) + f * (
                    cdya(M[α], M[β]) + cdya(M[β], M[α])
                )

        return d2WdCdC
