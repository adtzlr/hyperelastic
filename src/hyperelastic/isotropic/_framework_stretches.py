import numpy as np

from ..math import cdya, ddot, dya, eye, trace, eigh, tril_from_triu, transpose

class FrameworkStretch:
    r"""The Framework for a Total-Lagrangian stretch-based isotropic hyperelastic
    material formulation provides the material behaviour-independent parts for
    evaluating the second Piola-Kirchhoff stress tensor as well as its associated
    fourth-order elasticity tensor.

    The gradient as well as the hessian of the strain energy function are carried out
    w.r.t. the right Cauchy-Green deformation tensor. Hence, the work-conjugate stress
    tensor is one half of the second Piola-Kirchhoff stress tensor and the fourth-order
    elasticitiy tensor used here is a quarter of the Total-Lagrangian elasticity tensor.
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

        self.λ, self.M = eigh(C, fun=np.sqrt)

        self.dWdλ, statevars_new = self.material.gradient(self.λ, statevars)
        self.dWdλC = self.dWdλ / (2 * self.λ)

        dWdC = np.sum(np.expand_dims(self.dWdλC, axis=1) * self.M, axis=0)

        return dWdC, statevars_new

    def hessian(self, C, statevars):
        """The hessian as the second partial derivatives of the strain energy function
        w.r.t. the right Cauchy-Green deformation tensor (a quarter of the Lagrangian
        fourth-order elasticity tensor associated to the second Piola-Kirchhoff stress
        tensor)."""

        dWdC, statevars = self.gradient(C, statevars)
        d2Wdλdλ = self.material.hessian(
            self.λ, statevars
        )
        
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
            
            d2WdCdC += d2WdλC2[m] * dya(M[α], M[β])

            if β != α:
                v = λ[α]**2 - λ[β]**2
                mask = np.isclose(v, 0)

                w = np.zeros_like(v)
                w[~mask] = (dWdλC[α][~mask] - dWdλC[β][~mask]) / v[~mask]
                w[mask] = (d2WdλC2[β][mask] - d2WdλC2[m][mask]) / 2
                
                d2WdCdC += w * cdya(M[α], M[β])

        return d2WdCdC
