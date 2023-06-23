import felupe.math as fm
import numpy as np

from ...math import as_tensor, as_voigt, ddot, dot, eye, inv, trace


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
        self.C = fm.dot(fm.transpose(F), F)
        C = as_voigt(self.C)

        self.I1 = trace(C)
        self.I2 = (self.I1**2 - ddot(C, C)) / 2

        self.dWdI1, self.dWdI2, statevars_new = self.material.gradient(
            [self.I1, self.I2, statevars]
        )

        self.dI1dE = 2 * eye(C)
        self.dI2dE = self.I1 * self.dI1dE - 2 * C

        self.dI1dF = 2 * F
        self.dI2dF = 2 * (self.I1 * F - fm.dot(F, self.C))

        S = self.dWdI1 * self.dI1dE + self.dWdI2 * self.dI2dE

        return [fm.dot(F, as_tensor(S)), statevars_new]

    def hessian(self, x):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the deformation tensor ( fourth-order elasticity tensor)."""

        F = x[0]
        eye = fm.identity(F)

        dWdF, statevars = self.gradient(x)
        d2WdI1dI1, d2WdI1dI2, d2WdI2dI2 = self.material.hessian(
            [self.I1, self.I2, statevars]
        )

        d2I1dFdF = 2 * fm.cdya_ik(eye, eye, parallel=self.parallel)

        A = (
            d2WdI1dI1 * fm.dya(self.dI1dF, self.dI1dF, parallel=self.parallel)
            + self.dWdI1 * d2I1dFdF
        )

        if not np.allclose(d2WdI1dI2, 0):
            A += +d2WdI1dI2 * fm.dya(
                self.dI1dF, self.dI2dF, parallel=self.parallel
            ) + d2WdI1dI2 * fm.dya(self.dI2dF, self.dI1dF, parallel=self.parallel)

        if not np.allclose(self.dWdI2, 0):
            b = fm.dot(F, fm.transpose(F), parallel=self.parallel)

            eyeC = fm.cdya_ik(eye, self.C, parallel=self.parallel)
            beye = fm.cdya_ik(b, eye, parallel=self.parallel)

            dFFTFdF = eyeC + beye + fm.cdya_il(F, F, parallel=self.parallel)

            d2I2dFdF = (
                self.I1 * d2I1dFdF
                + 4 * fm.dya(F, F, parallel=self.parallel)
                - 2 * dFFTFdF
            )

            A += self.dWdI2 * d2I2dFdF + d2WdI2dI2 * fm.dya(
                self.dI2dF, self.dI2dF, parallel=self.parallel
            )

        return [A]
