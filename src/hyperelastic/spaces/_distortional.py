import felupe.math as fm
import numpy as np

from ..math import astensor, asvoigt, cdya, cdya_ik, ddot, dya, eye, inv, transpose


class DistortionalSpace:
    def __init__(self, material, parallel=False):
        self.parallel = parallel
        self.material = material

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [material.x[0], material.x[-1]]

    def gradient(self, x):
        F, statevars = x[0], x[-1]
        self.J = J = fm.det(F)
        self.C = C = asvoigt(fm.dot(fm.transpose(F), F))
        self.Cu = J ** (-2 / 3) * C
        dWudCu, statevars_new = self.material.gradient(self.Cu, statevars)

        self.Sb = Sb = 2 * J ** (-2 / 3) * dWudCu
        self.SbC = SbC = ddot(Sb, C)
        self.invC = invC = inv(C, determinant=J**2)

        self.S = Sb - SbC / 3 * invC

        return [fm.dot(F, astensor(self.S)), statevars_new]

    def hessian(self, x):
        F, statevars = x[0], x[-1]

        dWudF, statevars_new = self.gradient(x)
        I = eye(self.C)

        d2WdCdC = self.material.hessian(self.Cu, statevars)
        C4b = 4 * self.J ** (-4 / 3) * d2WdCdC

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
