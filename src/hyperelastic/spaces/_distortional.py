import felupe.math as fm
import numpy as np

from ..math import astensor, asvoigt, cdya, cdya_ik, ddot, dya, eye, inv, transpose, det


class DistortionalSpace:
    """The distortional (part of the deformation) space is a partial deformation with
    constant volume. For a given deformation map :math:`x(X)` and its deformation 
    gradient :math:`\boldsymbol{F}, the distortional part of the deformation gradient 
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
    evaluated by the help of its third invariant (the determinant).
    
    ..  math::

        \hat{\boldsymbol{C}} = I_3^{-1/3} \boldsymbol{C}
    
    The gradient as well as the hessian of the strain energy function are carried out
    w.r.t. the Green-Lagrange strain tensor. Hence, the work-conjugate stress
    tensor is the second Piola-Kirchhoff stress tensor and the fourth-order Total-
    Lagrangian elasticity tensor.
    
    ..  math::
        
        \boldsymbol{S}' = \frac{\partial \hat{\psi}}{\partial \frac{1}{2}\boldsymbol{C}}
    
    The (**physically** deviatoric) second Piola-Kirchhoff stress tensor is evaluated by
    the application of the chain rule.
    
    ..  math::
        
        \hat{\boldsymbol{S}} = \frac{\partial \hat{\psi}}
            {\partial \frac{1}{2}\hat{\boldsymbol{C}}}
    
    
    
    """
    def __init__(self, material, parallel=False):
        self.parallel = parallel
        self.material = material

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [material.x[0], material.x[-1]]

    def gradient(self, x):
        """The gradient as the partial derivative of the strain energy function w.r.t.
        the deformation gradient."""
        
        F, statevars = x[0], x[-1]
        self.C = C = asvoigt(fm.dot(fm.transpose(F), F))
        self.I3 = I3 = det(C)
        self.Cu = I3 ** (-1 / 3) * C
        dWudCu, statevars_new = self.material.gradient(self.Cu, statevars)

        self.Sb = Sb = 2 * I3 ** (-1 / 3) * dWudCu
        self.SbC = SbC = ddot(Sb, C)
        self.invC = invC = inv(C, determinant=I3)

        self.S = Sb - SbC / 3 * invC

        return [fm.dot(F, astensor(self.S)), statevars_new]

    def hessian(self, x):
        """The hessian as the second partial derivative of the strain energy function 
        w.r.t. the deformation gradient."""
        
        F, statevars = x[0], x[-1]

        dWudF, statevars_new = self.gradient(x)
        I = eye(self.C)

        d2WdCdC = self.material.hessian(self.Cu, statevars)
        C4b = 4 * self.I3 ** (-2 / 3) * d2WdCdC

        P4 = cdya(I, I) - dya(self.invC, self.C) / 3
        dinvCdC = -cdya(self.invC, self.invC)

        SbinvC = dya(self.Sb, self.invC)
        invCSb = transpose(SbinvC)
        invCinvC = dya(self.invC, self.invC)

        C4 = 2 / 3 * (-self.SbC * dinvCdC - SbinvC - invCSb + self.SbC / 3 * invCinvC)

        if not np.allclose(C4b, 0):
            C4 += ddot(ddot(P4, C4b, mode=(4, 4)), transpose(P4), mode=(4, 4))

        if self.parallel:
            from einsumt import einsumt

            einsum = einsumt
        else:
            einsum = np.einsum

        A4 = einsum("iI...,kK...,IJKL...->iJkL...", F, F, astensor(C4, 4))

        return [A4 + cdya_ik(I, self.S)]
