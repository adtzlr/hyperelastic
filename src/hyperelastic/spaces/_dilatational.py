import numpy as np

from ..math import astensor, asvoigt, cdya, cdya_ik, ddot, det, dya, eye, inv, trace
from ._space import Space


class Dilatational(Space):
    def __init__(self, material, parallel=False, finalize=True, force=None, area=0):
        self.material = material

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [material.x[0], material.x[-1]]

        super().__init__(parallel=parallel, finalize=finalize, force=force, area=area)

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
        return [self.piola(F=F, S=self.S, detF=np.sqrt(self.I3)), statevars_new]

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

        return [self.piola(F=F, S=self.S, detF=np.sqrt(self.I3), C4=C4, invC=self.invC)]
