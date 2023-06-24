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
        C = self.C
        invC = self.invC
        J = self.J
        S = self.S
        Sb = self.Sb
        SbC = self.SbC

        d2WdCdC = self.material.hessian(self.Cu, statevars)
        C4b = 4 * J ** (-4 / 3) * d2WdCdC
        I = eye(C)
        P4 = cdya(I, I) - dya(invC, C) / 3
        dinvCdC = -cdya(invC, invC)

        SbinvC = dya(Sb, invC)
        invCSb = transpose(SbinvC)

        C4 = 2 / 3 * (-SbC * dinvCdC - SbinvC - invCSb + SbC / 3 * dya(invC, invC))

        if not np.allclose(C4b, 0):
            C4 += ddot(ddot(P4, C4b, mode=(4, 4)), transpose(P4), mode=(4, 4))

        A4 = np.einsum("iI...,kK...,IJKL...->iJkL...", F, F, astensor(C4, 4))

        return [A4 + cdya_ik(I, S)]
