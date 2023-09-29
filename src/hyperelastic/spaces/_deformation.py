import numpy as np

from ..math import astensor, asvoigt, cdya_ik, eye
from ._space import Space


class Deformation(Space):
    r"""The deformation space.

    This class takes a Total-Lagrange material formulation and applies it on the
    deformation space.

    ..  math::

        \psi = \psi(\boldsymbol{C}(\boldsymbol{F}))

    The gradient of the strain energy function is carried out w.r.t. the Green Lagrange
    strain tensor. Hence, the work-conjugate stress tensor here refers to the second
    Piola-Kirchhoff stress tensor.

    ..  math::

        \boldsymbol{S} = \frac{\partial \psi}{\partial \frac{1}{2}\boldsymbol{C}}

    The hessian of the strain energy function is carried out w.r.t. the Green-Lagrange
    strain tensor. Hence, the work-conjugate elasticity tensor here refers to the
    fourth-order Total-Lagrangian elasticity tensor.

    ..  math::

        \mathbb{C} = \frac{\partial^2 \psi}{\partial \frac{1}{2}\boldsymbol{C}~
            \frac{1}{2}\boldsymbol{C}}


    Given a Total-Lagrange material formulation, for the variation and linearization of
    the virtual work of internal forces, the output quantities have to be transformed:
    The second Piola-Kirchhoff stress tensor is converted into the deformation gradient
    work-conjugate first Piola-Kirchhoff stress tensor, along with its fourth-order
    elasticity tensor. Also, the so-called geometric tangent stiffness component
    (initial stress matrix) is added to the fourth-order elasticity tensor.

    ..  math::

        \delta W_{int} &=
            - \int_V \boldsymbol{P} : \delta \boldsymbol{F} ~ dV

        \Delta \delta W_{int} &=
            - \int_V \delta \boldsymbol{F} : \mathbb{A} : \Delta \boldsymbol{F} ~ dV

    where

    ..  math::

        \boldsymbol{P} &= \boldsymbol{F} \boldsymbol{S}

        \mathbb{A}_{iJkL} &= F_{iI} F_{kK} \mathbb{C}_{IJKL} + \delta_{ik} S_{JL}

    """

    def __init__(self, material, parallel=False, finalize=True, force=None, area=0):
        self.material = material

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [material.x[0], material.x[-1]]

        super().__init__(parallel=parallel, finalize=finalize, force=force, area=area)

    def gradient(self, x):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the deformation gradient."""

        F, statevars = x[0], x[-1]

        self.C = asvoigt(self.einsum("kI...,kJ...->IJ...", F, F))
        dWdC, statevars_new = self.material.gradient(self.C, statevars)

        self.S = 2 * dWdC

        return [self.piola(F=F, S=self.S), statevars_new]

    def hessian(self, x):
        """The hessian as the second partial derivative of the strain energy function
        w.r.t. the deformation gradient."""

        F, statevars = x[0], x[-1]

        dWdF, statevars_new = self.gradient(x)
        d2WdCdC = self.material.hessian(self.C, statevars)

        return [self.piola(F=F, S=self.S, C4=4 * d2WdCdC)]
