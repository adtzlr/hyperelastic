import numpy as np

from ..math import cdya, ddot, det, dya, eye, inv, trace


class InvariantsFramework:
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

        I_3 &= \det(\boldsymbol{C})

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

        \psi_{,3} &= \frac{\partial \psi}{\partial I_3}

    and the partial derivatives of the invariants w.r.t. the right Cauchy-Green
    deformation tensor are defined.

    ..  math::

        \frac{\partial I_1}{\partial \boldsymbol{C}} &= \boldsymbol{I}

        \frac{\partial I_2}{\partial \boldsymbol{C}} &=
            \left( I_1 \boldsymbol{I} - \boldsymbol{C} \right)

        \frac{\partial I_3}{\partial \boldsymbol{C}} &= I_3 \boldsymbol{C}^{-1}

    The second Piola-Kirchhoff stress tensor is formulated by the application of the
    chain rule.

    ..  math::

        \frac{\partial \psi}{\partial \boldsymbol{C}} =
            \frac{\partial \psi}{\partial I_1}
            \frac{\partial I_1}{\partial \boldsymbol{C}}
            + \frac{\partial \psi}{\partial I_2}
            \frac{\partial I_2}{\partial \boldsymbol{C}}
            + \frac{\partial \psi}{\partial I_3}
            \frac{\partial I_3}{\partial \boldsymbol{C}}

    Furthermore, the second partial derivatives of the elasticity tensor are carried
    out.

    ..  math::

        \mathbb{C} &= \frac{\partial^2 \psi}{\partial I_1~\partial I_1}
            \left( \frac{\partial I_1}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_1}{\partial \boldsymbol{C}} \right)

            &+ \frac{\partial^2 \psi}{\partial I_2~\partial I_2}
            \left( \frac{\partial I_2}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_2}{\partial \boldsymbol{C}} \right)

            &+ \frac{\partial^2 \psi}{\partial I_3~\partial I_3}
            \left( \frac{\partial I_3}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_3}{\partial \boldsymbol{C}} \right)

            &+ \frac{\partial^2 \psi}{\partial I_1~\partial I_2}
            \left( \frac{\partial I_1}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_2}{\partial \boldsymbol{C}}
            + \frac{\partial I_2}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_1}{\partial \boldsymbol{C}} \right)

            &+ \frac{\partial^2 \psi}{\partial I_2~\partial I_3}
            \left( \frac{\partial I_2}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_3}{\partial \boldsymbol{C}}
            + \frac{\partial I_3}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_2}{\partial \boldsymbol{C}} \right)

            &+ \frac{\partial^2 \psi}{\partial I_1~\partial I_3}
            \left( \frac{\partial I_1}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_3}{\partial \boldsymbol{C}}
            + \frac{\partial I_3}{\partial \boldsymbol{C}} \otimes
            \frac{\partial I_1}{\partial \boldsymbol{C}} \right)

            &+ \frac{\partial \psi}{\partial I_1}
            \frac{\partial^2 I_1}{\partial \boldsymbol{C}~\partial \boldsymbol{C}}
            + \frac{\partial \psi}{\partial I_2}
            \frac{\partial^2 I_2}{\partial \boldsymbol{C}~\partial \boldsymbol{C}}
            + \frac{\partial \psi}{\partial I_3}
            \frac{\partial^2 I_3}{\partial \boldsymbol{C}~\partial \boldsymbol{C}}

    The only non material behaviour-related terms which are not already defined during
    stress evaluation are the second partial derivatives of the invariants w.r.t.
    the right Cauchy-Green deformation tensor.

    ..  math::

        \frac{\partial^2 I_12}{\partial \boldsymbol{C}~\partial \boldsymbol{C}} &=
            \mathbb{0}

        \frac{\partial^2 I_2}{\partial \boldsymbol{C}~\partial \boldsymbol{C}} &=
            \boldsymbol{I} \otimes \boldsymbol{I} - \boldsymbol{I} \odot \boldsymbol{I}

        \frac{\partial^2 I_3}{\partial \boldsymbol{C}~\partial \boldsymbol{C}} &=
            I_3 \left( \boldsymbol{C}^{-1} \otimes \boldsymbol{C}^{-1}
                - \boldsymbol{C}^{-1} \odot \boldsymbol{C}^{-1}
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

        self.I = I = eye(C)

        self.I1 = I1 = trace(C)
        self.I2 = (I1**2 - ddot(C, C)) / 2
        self.I3 = I3 = det(C)

        self.dWdI1, self.dWdI2, self.dWdI3, statevars_new = self.material.gradient(
            self.I1, self.I2, self.I3, statevars
        )

        ntrax = len(C.shape[1:])
        dWdC = np.zeros((6, *np.ones(ntrax, dtype=int)))

        if self.dWdI1 is not None:
            self.dI1dC = I
            dWdC = dWdC + self.dWdI1 * self.dI1dC

        if self.dWdI2 is not None:
            self.dI2dC = I1 * I - C
            dWdC = dWdC + self.dWdI2 * self.dI2dC

        if self.dWdI3 is not None:
            self.invC = invC = inv(C, determinant=self.I3)
            self.dI3dC = I3 * invC
            dWdC = dWdC + self.dWdI3 * self.dI3dC

        return dWdC, statevars_new

    def hessian(self, C, statevars):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the right Cauchy-Green deformation tensor (a quarter of the Lagrangian
        fourth-order elasticity tensor associated to the second Piola-Kirchhoff stress
        tensor)."""

        dWdC, statevars = self.gradient(C, statevars)
        (
            d2WdI1dI1,
            d2WdI2dI2,
            d2WdI3dI3,
            d2WdI1dI2,
            d2WdI2dI3,
            d2WdI1dI3,
        ) = self.material.hessian(self.I1, self.I2, self.I3, statevars)
        I = self.I

        ntrax = len(C.shape[1:])
        d2WdCdC = np.zeros((6, 6, *np.ones(ntrax, dtype=int)))

        if self.dWdI1 is not None:
            dI1dC = self.dI1dC
            # d2I1dCdC = 0
            # d2WdCdC = d2WdCdC + self.dWdI1 * d2I1dCdC

        if self.dWdI2 is not None:
            dI2dC = self.dI2dC
            d2I2dCdC = dya(I, I) - cdya(I, I)
            d2WdCdC = d2WdCdC + self.dWdI2 * d2I2dCdC

        if self.dWdI3 is not None:
            dI3dC = self.dI3dC
            invC = self.invC
            d2I3dCdC = self.I3 * (dya(invC, invC) - cdya(invC, invC))
            d2WdCdC = d2WdCdC + self.dWdI3 * d2I3dCdC

        if d2WdI1dI1 is not None:
            d2WdCdC = d2WdCdC + d2WdI1dI1 * dya(dI1dC, dI1dC)

        if d2WdI2dI2 is not None:
            d2WdCdC = d2WdCdC + d2WdI2dI2 * dya(dI2dC, dI2dC)

        if d2WdI3dI3 is not None:
            d2WdCdC = d2WdCdC + d2WdI2dI2 * dya(dI2dC, dI2dC)

        if d2WdI1dI2 is not None:
            d2WdCdC = d2WdCdC + d2WdI1dI2 * (dya(dI1dC, dI2dC) + dya(dI2dC, dI1dC))

        if d2WdI2dI3 is not None:
            d2WdCdC = d2WdCdC + d2WdI2dI3 * (dya(dI2dC, dI3dC) + dya(dI3dC, dI2dC))

        if d2WdI1dI3 is not None:
            d2WdCdC = d2WdCdC + d2WdI1dI3 * (dya(dI1dC, dI3dC) + dya(dI3dC, dI1dC))

        return d2WdCdC
