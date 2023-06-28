# from felupe.math import cdya_il, ddot, det, dya, identity, inv, trace, transpose
import felupe.math as fm
import numpy as np

from ..math import astensor, asvoigt, cdya, cdya_ik, ddot, det, dya, eye, inv, transpose, trace


class DilatationalSpace:
    def __init__(self, material, parallel=False):
        self.parallel = parallel
        self.material = material

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [material.x[0], material.x[-1]]

    def gradient(self, x):
        
        F, statevars = x[0], x[-1]
        self.C = C = asvoigt(fm.dot(fm.transpose(F), F))
        self.I3 = I3 = det(C)
        self.Co = I3 ** (1 / 3) * C
        dWodCo, statevars_new = self.material.gradient(self.Co, statevars)

        self.Sb = Sb = 2 * I3 ** (1 / 3) * dWodCo
        self.tr_Sb = tr_Sb = trace(Sb)
        self.invC = invC = inv(C, determinant=I3)

        self.S = I3 ** (1 / 3) / 3 * tr_Sb * invC

        return [fm.dot(F, astensor(self.S)), statevars_new]

    def hessian(self, x):
        
        # work in progress...
        
        F, statevars = x[0], x[-1]

        dWodF, statevars_new = self.gradient(x)
        I = eye(self.C)

        d2WodCodCo = self.material.hessian(self.Co, statevars)
        C4b = 4 * self.I3 ** (2 / 3) * d2WodCodCo

        P4 = cdya(I, I) - dya(self.invC, self.C) / 3
        I4 = cdya(self.invC, self.invC)

        SbinvC = dya(self.Sb, self.invC)
        invCSb = transpose(SbinvC)
        invCinvC = dya(self.invC, self.invC)

        C4 = 2 / 3 * (self.SbC * I4 - SbinvC - invCSb + self.SbC / 3 * invCinvC)

        if not np.allclose(C4b, 0):
            C4 += ddot(ddot(P4, C4b, mode=(4, 4)), transpose(P4), mode=(4, 4))

        if self.parallel:
            from einsumt import einsumt

            einsum = einsumt
        else:
            einsum = np.einsum

        A4 = einsum("iI...,kK...,IJKL...->iJkL...", F, F, astensor(C4, 4))

        return [A4 + cdya_ik(I, self.S)]
    
        # statevars = x[-1]
        # self.P, statevars_new = self.gradient(x)

        # self.Ao = self.material.hessian([self.Fo, statevars])[0]
        # self.Av = self.J ** (-2 / 3) * self.Ao
        # self.IAv = ddot(self.eye, self.Av, mode=(2, 4), parallel=self.parallel)
        # self.IAvI = ddot(self.IAv, self.eye, mode=(2, 2), parallel=self.parallel)
        # self.iFTiFT = dya(self.iFT, self.iFT, parallel=self.parallel)
        # self.diFTdF = -cdya_il(self.iFT, self.iFT, parallel=self.parallel)

        # A = (self.tr_Pv + self.IAvI) / 9 * self.iFTiFT + self.tr_Pv / 3 * self.diFTdF

        # return [A]
