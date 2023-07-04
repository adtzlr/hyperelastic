import numpy as np

from ..math import astensor, asvoigt, cdya, cdya_ik, ddot, det, dya, eye, inv, transpose


class Distortional:
    r"""The distortional (part of the deformation) space is a partial deformation with
    constant volume. For a given deformation map :math:`x(X)` and its deformation
    gradient :math:`\boldsymbol{F}`, the distortional part of the deformation gradient
    :math:`\hat{\boldsymbol{F}}` is obtained by a multiplicative (consecutive) split
    into a volume-changing (dilatational) and a constant-volume (distortional) part of
    the deformation gradient. Due to the fact that the dilatational part is proportional
    to the unit tensor, the order of these partial deformations is not unique.

    ..  math::

        \boldsymbol{F} = \overset{\circ}{\boldsymbol{F}} \hat{\boldsymbol{F}} =
            \hat{\boldsymbol{F}} \overset{\circ}{\boldsymbol{F}}

    This class takes a Total-Lagrange material formulation and applies it only on the
    distortional space.

    ..  math::

        \hat{\psi} = \psi(\hat{\boldsymbol{C}}(\boldsymbol{F}))

    The distortional (unimodular) part of the right Cauchy-Green deformation tensor is
    evaluated by the help of its third invariant (the determinant). The determinant of
    a distortional (an unimodular) tensor equals to one.

    ..  math::

        \hat{\boldsymbol{C}} = I_3^{-1/3} \boldsymbol{C}

    The gradient of the strain energy function is carried out w.r.t. the Green Lagrange
    strain tensor. Hence, the work-conjugate stress tensor used in this space
    projection refers to the second Piola-Kirchhoff stress tensor.

    ..  math::

        \boldsymbol{S}' = \frac{\partial \hat{\psi}}{\partial \frac{1}{2}\boldsymbol{C}}

    The distortional space projection leads to a **physically** deviatoric second
    Piola-Kirchhoff stress tensor, evaluated by the application of the chain rule.

    ..  math::

        \hat{\boldsymbol{S}} = \frac{\partial \hat{\psi}}
            {\partial \frac{1}{2}\hat{\boldsymbol{C}}}

    The (**phyiscally**) deviatoric projection is obtained by the partial derivative of
    the distortional part of the right Cauchy-Green deformation tensor w.r.t. the right
    Cauchy-Green deformation tensor.

    ..  math::

        \frac{\partial \hat{\boldsymbol{C}}}{\partial \boldsymbol{C}} =
            \frac{\partial I_3^{-1/3} \boldsymbol{C}}{\partial \boldsymbol{C}} =
            I_3^{-1/3} \left( \boldsymbol{1} \odot \boldsymbol{1}
            - \frac{1}{3} \boldsymbol{C} \otimes \boldsymbol{C}^{-1} \right)

    This partial derivative is used to perform the distortional space projection of the
    second Piola-Kirchhoff stress tensor. Instead of asserting the determinant-scaling
    to the fourth-order projection tensor, this factor is combined with the second
    Piola-Kirchhoff stress tensor in the distortional space. Hence, the stress tensor in
    the distortional space, scaled by :math:`I_3^{-1/3}`, is introduced as a new
    (frequently re-used) variable, denoted by an overset bar.

    ..  math::

        \boldsymbol{S}' &= \mathbb{P} : \bar{\boldsymbol{S}}

        \bar{\boldsymbol{S}} &= I_3^{-1/3} \hat{\boldsymbol{S}}

        \mathbb{P} &= \boldsymbol{1} \odot \boldsymbol{1}
            - \frac{1}{3} \boldsymbol{C}^{-1} \otimes \boldsymbol{C}

    The evaluation of the double-dot product for the distortional space projection leads
    to the mathematical deviator of the product between the scaled distortional space
    stress tensor and the right Cauchy-Green deformation tensor, right multiplied by the
    inverse of the right Cauchy-Green deformation tensor.

    ..  math::

        \boldsymbol{S}' = \bar{\boldsymbol{S}}
            - \frac{\bar{\boldsymbol{S}}:\boldsymbol{C}}{3} \boldsymbol{C}^{-1}
            = \text{dev}(\bar{\boldsymbol{S}} \boldsymbol{C}) \boldsymbol{C}^{-1}

    The hessian of the strain energy function is carried out w.r.t. the Green-Lagrange
    strain tensor. Hence, the work-conjugate stress tensor used in this space
    projection refers to the fourth-order Total-Lagrangian elasticity tensor.

    ..  math::

        \mathbb{C}' = \frac{\partial^2 \hat{\psi}}{\partial \frac{1}{2}\boldsymbol{C}~
            \frac{1}{2}\boldsymbol{C}}

    The evaluation of this second partial derivative leads to the elasticity tensor of
    the distortional space projection. The remaining determinant scaling terms of the
    projection tensor are included in the determinant-modified fourth-order elasticity
    tensor, denoted with an overset bar.

    ..  math::

        \mathbb{C}' = \mathbb{P} : \bar{\mathbb{C}} : \mathbb{P}^T + \frac{2}{3} \left(
            \left( \bar{\boldsymbol{S}}:\boldsymbol{C} \right)
            \boldsymbol{C}^{-1} \odot \boldsymbol{C}^{-1}
            - \bar{\boldsymbol{S}} \otimes \boldsymbol{C}^{-1}
            - \boldsymbol{C}^{-1} \otimes \bar{\boldsymbol{S}}
            + \frac{1}{3} \left( \bar{\boldsymbol{S}}:\boldsymbol{C} \right)
            \boldsymbol{C}^{-1} \otimes \boldsymbol{C}^{-1}
        \right)

    ..  math::

        \bar{\mathbb{C}} = I_3^{-2/3} \hat{\mathbb{C}}

    For the variation and linearization of the virtual work of internal forces, the
    output quantities have to be transformed: The second Piola-Kirchhoff stress tensor
    is converted into the  deformation gradient work-conjugate first Piola-Kirchhoff
    stress tensor, along with its fourth-order elasticity tensor. Also, the so-called
    geometric tangent stiffness component (initial stress matrix) is added to the
    fourth-order elasticity tensor.

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

    def __init__(self, material, parallel=False):
        self.parallel = parallel
        self.material = material

        if self.parallel:
            from einsumt import einsumt

            self.einsum = einsumt
        else:
            self.einsum = np.einsum

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [material.x[0], material.x[-1]]

    def gradient(self, x):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the deformation gradient."""

        F, statevars = x[0], x[-1]

        self.C = C = asvoigt(self.einsum("kI...,kJ...->IJ...", F, F))
        self.I3 = I3 = det(C)
        self.Cu = I3 ** (-1 / 3) * C
        dWudCu, statevars_new = self.material.gradient(self.Cu, statevars)

        self.Sb = Sb = 2 * I3 ** (-1 / 3) * dWudCu
        self.SbC = SbC = ddot(Sb, C)
        self.invC = invC = inv(C, determinant=I3)

        self.S = Sb - SbC / 3 * invC

        return [self.einsum("iK...,KJ...->iJ...", F, astensor(self.S)), statevars_new]

    def hessian(self, x):
        """The hessian as the second partial derivative of the strain energy function
        w.r.t. the deformation gradient."""

        F, statevars = x[0], x[-1]

        dWudF, statevars_new = self.gradient(x)
        I = eye(self.C)

        d2WdCdC = self.material.hessian(self.Cu, statevars)
        C4b = 4 * self.I3 ** (-2 / 3) * d2WdCdC

        P4 = cdya(I, I) - dya(self.invC, self.C) / 3
        I4 = cdya(self.invC, self.invC)

        SbinvC = dya(self.Sb, self.invC)
        invCSb = transpose(SbinvC)
        invCinvC = dya(self.invC, self.invC)

        C4 = 2 / 3 * (self.SbC * I4 - SbinvC - invCSb + self.SbC / 3 * invCinvC)

        if not np.allclose(C4b, 0):
            C4 += ddot(ddot(P4, C4b, mode=(4, 4)), transpose(P4), mode=(4, 4))

        A4 = self.einsum("iI...,kK...,IJKL...->iJkL...", F, F, astensor(C4, 4))

        return [A4 + cdya_ik(I, self.S)]
