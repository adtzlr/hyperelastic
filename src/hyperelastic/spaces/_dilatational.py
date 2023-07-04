import numpy as np

from ..math import astensor, asvoigt, cdya, cdya_ik, ddot, det, dya, eye, inv, trace


class Dilatational:
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
        F, statevars = x[0], x[-1]

        self.C = asvoigt(self.einsum("kI...,kJ...->IJ...", F, F))
        self.I3 = det(self.C)
        self.Co = self.I3 ** (1 / 3) * self.C
        dWodCo, statevars_new = self.material.gradient(self.Co, statevars)

        self.Sb = 2 * self.I3 ** (1 / 3) * dWodCo
        self.tr_Sb = trace(self.Sb)
        self.invC = inv(self.C, determinant=self.I3)

        self.S = self.tr_Sb / 3 * self.invC
        return [self.einsum("iK...,KJ...->iJ...", F, astensor(self.S)), statevars_new]

    def hessian(self, x):
        F, statevars = x[0], x[-1]

        dWodF, statevars_new = self.gradient(x)

        d2WodCodCo = self.material.hessian(self.Co, statevars)
        C4b = 4 * self.I3 ** (2 / 3) * d2WodCodCo

        I = eye(self.C)
        tr_C4b = ddot(ddot(I, C4b, mode=(2, 4)), I)
        I4 = cdya(self.invC, self.invC)

        C4 = (2 * self.tr_Sb + tr_C4b) * dya(
            self.invC, self.invC
        ) / 9 - 2 / 3 * self.tr_Sb * I4

        A4 = self.einsum("iI...,kK...,IJKL...->iJkL...", F, F, astensor(C4, 4))

        return [A4 + cdya_ik(I, self.S)]
