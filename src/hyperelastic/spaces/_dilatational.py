from felupe.math import cdya_il, ddot, det, dya, identity, inv, trace, transpose


class DilatationalSpace:
    def __init__(self, material, parallel=False):
        self.parallel = parallel
        self.material = material

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [material.x[0], material.x[-1]]

    def gradient(self, x):
        F, statevars = x[0], x[-1]
        self.J = det(F)
        self.eye = identity(F)
        self.Fo = self.J ** (1 / 3) * self.eye
        self.Po, statevars_new = self.material.gradient([self.Fo, statevars])
        self.Pv = self.J ** (1 / 3) * self.Po
        self.tr_Pv = trace(self.Pv)
        self.iFT = transpose(inv(F, determinant=self.J))

        return [self.tr_Pv / 3 * self.iFT, statevars_new]

    def hessian(self, x):
        statevars = x[-1]
        self.P, statevars_new = self.gradient(x)

        self.Ao = self.material.hessian([self.Fo, statevars])[0]
        self.Av = self.J ** (-2 / 3) * self.Ao
        self.IAv = ddot(self.eye, self.Av, mode=(2, 4), parallel=self.parallel)
        self.IAvI = ddot(self.IAv, self.eye, mode=(2, 2), parallel=self.parallel)
        self.iFTiFT = dya(self.iFT, self.iFT, parallel=self.parallel)
        self.diFTdF = -cdya_il(self.iFT, self.iFT, parallel=self.parallel)

        A = (self.tr_Pv + self.IAvI) / 9 * self.iFTiFT + self.tr_Pv / 3 * self.diFTdF

        return [A]
