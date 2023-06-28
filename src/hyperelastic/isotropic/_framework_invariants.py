import numpy as np

from ..math import cdya, ddot, dya, eye, trace


class FrameworkInvariants:
    r"""The Framework for a Total-Lagrangian invariant-based isotropic hyperelastic
    material formulation provides the material behaviour-independent parts for
    evaluating the second Piola-Kirchhoff stress tensor as well as its associated
    fourth-order elasticity tensor.

    The gradient as well as the hessian of the strain energy function are carried out
    w.r.t. the right Cauchy-Green deformation tensor. Hence, the work-conjugate stress
    tensor is one half of the second Piola-Kirchhoff stress tensor and the fourth-order
    elasticitiy tensor used here is a quarter of the Total-Lagrangian elasticity tensor.

    ..  math::

        \psi(\boldsymbol{C}) =
            \psi(I_1(\boldsymbol{C}), I_2(\boldsymbol{C}), I_3(\boldsymbol{C}))

    The first and second invariants of the left or right Cauchy-Green deformation tensor
    are identified as factors of their characteristic polynomial,

    ..  math::

        I_1 &= \text{tr}(\boldsymbol{C})

        I_2 &= \frac{1}{2}
            \left( \text{tr}(\boldsymbol{C})^2 - \text{tr}(\boldsymbol{C}^2) \right)

    where the Cauchy-Green deformation tensors eliminate the rigid body rotations
    of the deformation gradient and serve as a quadratic change-of-length measure of
    the deformation.

    ..  math::

        \boldsymbol{C} &= \boldsymbol{F}^T \boldsymbol{F}

        \boldsymbol{b} &= \boldsymbol{F} \boldsymbol{F}^T

    The first partial derivatives of the strain energy function w.r.t. the invariants

    ..  math::

        \psi_{,1} &= \frac{\partial \psi}{\partial I_1}

        \psi_{,2} &= \frac{\partial \psi}{\partial I_2}

    and the partial derivatives of the invariants w.r.t. the right Cauchy-Green
    deformation tensor are defined.

    ..  math::

        \frac{\partial I_1}{\partial \boldsymbol{C}} &= 2 \boldsymbol{1}

        \frac{\partial I_2}{\partial \boldsymbol{C}} &=
            2 \left( I_1 \boldsymbol{1} - \boldsymbol{C} \right)

    The second Piola-Kirchhoff stress tensor is formulated by the application of the
    chain rule.

    ..  math::

        \frac{\partial \psi}{\partial \boldsymbol{C}} =
            \frac{\partial \psi}{\partial I_1}
            \frac{\partial I_1}{\partial \boldsymbol{C}}
            + \frac{\partial \psi}{\partial I_2}
            \frac{\partial I_2}{\partial \boldsymbol{C}}

    Furthermore, the second partial derivatives of the elasticity tensor are carried
    out.

    ..  math::

        \mathbb{C} &= \frac{\partial^2 \psi}{\partial I_1~\partial I_1}
            \left( \frac{\partial I_1}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_1}{\partial \boldsymbol{C}} \right)
            + \frac{\partial^2 \psi}{\partial I_2~\partial I_2}
            \left( \frac{\partial I_2}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_2}{\partial \boldsymbol{C}} \right)

            &+ \frac{\partial^2 \psi}{\partial I_1~\partial I_2}
            \left( \frac{\partial I_1}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_2}{\partial \boldsymbol{C}}
            + \frac{\partial I_2}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_1}{\partial \boldsymbol{C}} \right)

            &+ \frac{\partial \psi}{\partial I_1}
            \frac{\partial^2 I_1}{\partial \boldsymbol{C}~\partial \boldsymbol{C}}
            + \frac{\partial \psi}{\partial I_2}
            \frac{\partial^2 I_2}{\partial \boldsymbol{C}~\partial \boldsymbol{C}}

    The only non material behaviour-related term which is not already defined during
    stress evaluation is the second partial derivatives of the second invariant w.r.t.
    the right Cauchy-Green deformation tensor.

    ..  math::

        \frac{\partial^2 I_2}{\partial \boldsymbol{C}~\partial \boldsymbol{C}} &=
            \boldsymbol{I} \otimes \boldsymbol{I} - \boldsymbol{I} \odot \boldsymbol{I}

    """

    def __init__(self, material, parallel=False):
        """Initialize the Framework for an invariant-based isotropic hyperelastic
        material formulation."""

        self.parallel = parallel
        self.material = material

        self.x = [np.eye(3), np.zeros(0)]
        if hasattr(self.material, "x"):
            self.x = [self.material.x[0], self.material.x[-1]]

    def gradient(self, C, statevars):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the right Cauchy-Green deformation tensor (one half of the second Piola
        Kirchhoff stress tensor)."""

        self.I = I = eye(C)

        self.I1 = I1 = trace(C)
        self.I2 = (I1**2 - ddot(C, C)) / 2

        self.dWdI1, self.dWdI2, statevars_new = self.material.gradient(
            self.I1, self.I2, statevars
        )

        self.dI1dC = I
        self.dI2dC = I1 * I - C

        dWdC = self.dWdI1 * self.dI1dC + self.dWdI2 * self.dI2dC

        return dWdC, statevars_new

    def hessian(self, C, statevars):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the right Cauchy-Green deformation tensor (a quarter of the Lagrangian
        fourth-order elasticity tensor associated to the second Piola-Kirchhoff stress
        tensor)."""

        dWdE, statevars = self.gradient(C, statevars)
        d2WdI1dI1, d2WdI2dI2, d2WdI1dI2 = self.material.hessian(
            self.I1, self.I2, statevars
        )
        I = self.I
        dI1dC = self.dI1dC
        dI2dC = self.dI2dC

        ntrax = len(C.shape[1:])
        d2WdCdC = np.zeros((6, 6, *np.ones(ntrax, dtype=int)))

        if not np.allclose(d2WdI1dI1, 0):
            d2WdCdC = d2WdCdC + d2WdI1dI1 * dya(dI1dC, dI1dC)

        if not np.allclose(d2WdI1dI2, 0):
            d2WdCdC = d2WdCdC + d2WdI1dI2 * (dya(dI1dC, dI2dC) + dya(dI2dC, dI1dC))

        if not np.allclose(self.dWdI2, 0):
            d2I2dCdC = dya(I, I) - cdya(I, I)

            d2WdCdC = d2WdCdC + self.dWdI2 * d2I2dCdC + d2WdI2dI2 * dya(dI2dC, dI2dC)

        return d2WdCdC
