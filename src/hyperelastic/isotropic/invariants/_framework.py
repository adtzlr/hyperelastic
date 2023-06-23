import felupe.math as fm
import numpy as np

from ...math import (
    as_tensor,
    as_voigt,
    cdya,
    cdya_ik,
    ddot,
    dot,
    dya,
    eye,
    inv,
    piola,
    trace,
)


class Framework:
    r"""The Framework for an invariant-based isotropic hyperelastic material
    formulation provides the material behaviour-independent parts for evaluating
    the first Piola-Kirchhoff stress tensor as well as its associated fourth-order
    elasticity tensor.

    ..  math::

        \psi(\boldsymbol{F}) =
            \psi(I_1(\boldsymbol{F}), I_2(\boldsymbol{F}), I_3(\boldsymbol{F}))

    The first and second invariants of the left or right Cauchy-Green deformation tensor

    ..  math::

        I_1 &= \text{tr}(\boldsymbol{C})

        I_2 &= \frac{1}{2}
            \left( \text{tr}(\boldsymbol{C})^2 - \text{tr}(\boldsymbol{C}^2) \right)

    where the right Cauchy-Green deformation tensor eliminates the rigid body rotations
    of the deformation gradient and serves as a quadratic change-of-length measure of
    the deformation.

    ..  math::

        \boldsymbol{C} &= \boldsymbol{F}^T \boldsymbol{F}

        \boldsymbol{b} &= \boldsymbol{F} \boldsymbol{F}^T

    The first partial derivatives of the strain energy function w.r.t. the invariants

    ..  math::

        \psi_{,1} &= \frac{\partial \psi}{\partial I_1}

        \psi_{,2} &= \frac{\partial \psi}{\partial I_2}

    and the partial derivatives of the invariants w.r.t. the deformation
    gradient are defined.

    ..  math::

        \frac{\partial I_1}{\partial \boldsymbol{F}} &= 2 \boldsymbol{F}

        \frac{\partial I_2}{\partial \boldsymbol{F}} &=
            2 \left( I_1 \boldsymbol{F} - \boldsymbol{F} \boldsymbol{C} \right)

    The first Piola-Kirchhoff stress tensor is formulated by the application of the
    chain rule.

    ..  math::

        \boldsymbol{P} = \frac{\partial \psi}{\partial I_1}
            \frac{\partial I_1}{\partial \boldsymbol{F}}
            + \frac{\partial \psi}{\partial I_2}
            \frac{\partial I_2}{\partial \boldsymbol{F}}

    Furthermore, the second partial derivatives of the elasticity tensor are carried
    out.

    ..  math::

        \mathbb{A} &= \frac{\partial^2 \psi}{\partial I_1~\partial I_1}
            \left( \frac{\partial I_1}{\partial \boldsymbol{F}} \otimes
            \frac{\partial I_1}{\partial \boldsymbol{F}} \right)
            + \frac{\partial^2 \psi}{\partial I_2~\partial I_2}
            \left( \frac{\partial I_2}{\partial \boldsymbol{F}} \otimes
            \frac{\partial I_2}{\partial \boldsymbol{F}} \right)

            &+ \frac{\partial^2 \psi}{\partial I_1~\partial I_2}
            \left( \frac{\partial I_1}{\partial \boldsymbol{F}} \otimes
            \frac{\partial I_2}{\partial \boldsymbol{F}}
            + \frac{\partial I_2}{\partial \boldsymbol{F}} \otimes
            \frac{\partial I_1}{\partial \boldsymbol{F}} \right)

            &+ \frac{\partial \psi}{\partial I_1}
            \frac{\partial^2 I_1}{\partial \boldsymbol{F}~\partial \boldsymbol{F}}
            + \frac{\partial \psi}{\partial I_2}
            \frac{\partial^2 I_2}{\partial \boldsymbol{F}~\partial \boldsymbol{F}}

    The only non material behaviour-related terms which are not already defined are
    the second partial derivatives of the invariants w.r.t. the deformation gradient.

    ..  math::

        \frac{\partial^2 I_1}{\partial \boldsymbol{F}~\partial \boldsymbol{F}} &=
            2~\boldsymbol{I} \overset{\small{ik}}{\odot} \boldsymbol{I}

        \frac{\partial^2 I_2}{\partial \boldsymbol{F}~\partial \boldsymbol{F}} &=
            I_1 \frac{\partial^2 I_1}{\partial \boldsymbol{F}~\partial \boldsymbol{F}}
            + 4~\boldsymbol{F} \otimes \boldsymbol{F}
            - 2~\frac{\partial \boldsymbol{F} \boldsymbol{C}}{\partial \boldsymbol{F}}

        \frac{\partial \boldsymbol{F} \boldsymbol{C}}{\partial \boldsymbol{F}} &=
            \boldsymbol{I} \overset{\small{ik}}{\odot} \boldsymbol{C}
            + \boldsymbol{b} \overset{\small{ik}}{\odot} \boldsymbol{I}
            + \boldsymbol{F} \overset{\small{il}}{\odot} \boldsymbol{F}

    """

    def __init__(self, material, parallel=False):
        """Initialize the Framework for an invariant-based isotropic hyperelastic
        material formulation."""

        self.parallel = parallel
        self.material = material

        self.x = [np.eye(3), np.zeros(0)]
        if hasattr(self.material, "x"):
            self.x = [self.material.x[0], self.material.x[-1]]

    def gradient(self, x):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the deformation gradient (first Piola Kirchhoff stress tensor)."""

        F, statevars = x[0], x[-1]
        self.C = as_voigt(fm.dot(fm.transpose(F), F))
        self.I = eye(self.C)

        self.I1 = trace(self.C)
        self.I2 = (self.I1**2 - ddot(self.C, self.C)) / 2

        self.dWdI1, self.dWdI2, statevars_new = self.material.gradient(
            [self.I1, self.I2, statevars]
        )

        self.dI1dE = 2 * self.I
        self.dI2dE = self.I1 * self.dI1dE - 2 * self.C

        self.S = self.dWdI1 * self.dI1dE + self.dWdI2 * self.dI2dE

        return [fm.dot(F, as_tensor(self.S)), statevars_new]

    def hessian(self, x):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the deformation tensor ( fourth-order elasticity tensor)."""

        F = x[0]

        dWdF, statevars = self.gradient(x)
        d2WdI1dI1, d2WdI1dI2, d2WdI2dI2 = self.material.hessian(
            [self.I1, self.I2, statevars]
        )

        C4 = d2WdI1dI1 * dya(self.dI1dE, self.dI1dE)

        if not np.allclose(d2WdI1dI2, 0):
            C4 += d2WdI1dI2 * dya(self.dI1dE, self.dI2dE) + d2WdI1dI2 * dya(
                self.dI2dE, self.dI1dE
            )

        if not np.allclose(self.dWdI2, 0):
            d2I2dEdE = 4 * (dya(self.I, self.I) - cdya(self.I, self.I))

            C4 += self.dWdI2 * d2I2dEdE + d2WdI2dI2 * dya(self.dI2dE, self.dI2dE)

        A = np.einsum("iI...,kK...,IJKL...->iJkL...", F, F, as_tensor(C4, 4))

        return [A + cdya_ik(self.I, self.S)]
