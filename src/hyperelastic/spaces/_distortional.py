from felupe.math import cdya_il, ddot, det, dya, inv, transpose


class DistortionalSpace:
    def __init__(self, material, parallel=False):
        self.parallel = parallel
        self.material = material

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [material.x[0], material.x[-1]]

    def gradient(self, x):
        F, statevars = x[0], x[-1]
        self.J = det(F)
        self.Fu = self.J ** (-1 / 3) * F
        self.Pu, statevars_new = self.material.gradient([self.Fu, statevars])
        self.Pb = self.J ** (-1 / 3) * self.Pu
        self.PbF = ddot(self.Pb, F, parallel=self.parallel)
        self.iFT = transpose(inv(F, determinant=self.J))

        return [self.Pb - self.PbF / 3 * self.iFT, statevars_new]

    def hessian(self, x):
        F, statevars = x[0], x[-1]
        self.P, statevars_new = self.gradient(x)

        self.Au = self.material.hessian([self.Fu, statevars])[0]
        self.iFTiFT = dya(self.iFT, self.iFT, parallel=self.parallel)

        self.Ab = self.J ** (-2 / 3) * self.Au
        self.AbF = ddot(self.Ab, F, mode=(4, 2), parallel=self.parallel)
        self.FAb = ddot(F, self.Ab, mode=(2, 4), parallel=self.parallel)
        self.FAbF = ddot(F, self.AbF, mode=(2, 2), parallel=self.parallel)

        return [
            self.Ab
            - dya(self.Pb, self.iFT, parallel=self.parallel) / 3
            - dya(self.iFT, self.Pb, parallel=self.parallel) / 3
            - dya(self.AbF, self.iFT, parallel=self.parallel) / 3
            - dya(self.iFT, self.FAb, parallel=self.parallel) / 3
            + self.PbF / 3 * cdya_il(self.iFT, self.iFT, parallel=self.parallel)
            + (self.PbF + self.FAbF) / 9 * self.iFTiFT
        ]
