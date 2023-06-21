from felupe.math import cdya_ik, cdya_il, ddot, dot, dya, identity, transpose


class Framework:
    def __init__(self, material, parallel=False):
        self.parallel = parallel
        self.material = material

        self.x = [self.material.x[0], self.material.x[-1]]

    def gradient(self, x):
        F, statevars = x[0], x[-1]
        self.C = dot(transpose(F), F, parallel=self.parallel)

        self.I1 = ddot(F, F, parallel=self.parallel)
        self.I2 = (self.I1**2 - ddot(self.C, self.C, parallel=self.parallel)) / 2

        self.FC = dot(F, self.C, parallel=self.parallel)

        self.dWdI1, self.dWdI2, statevars_new = self.material.gradient(
            [self.I1, self.I2, statevars]
        )

        self.dI1dF = 2 * F
        self.dI2dF = 2 * (self.I1 * F - self.FC)

        return [self.dWdI1 * self.dI1dF + self.dWdI2 * self.dI2dF, statevars_new]

    def hessian(self, x):
        F = x[0]
        eye = identity(F)

        dWdF, statevars = self.gradient(x)
        d2WdI1dI1, d2WdI1dI2, d2WdI2dI2 = self.material.hessian(
            [self.I1, self.I2, statevars]
        )

        b = dot(F, transpose(F), parallel=self.parallel)

        eyeC = cdya_ik(eye, self.C, parallel=self.parallel)
        beye = cdya_ik(b, eye, parallel=self.parallel)

        dFFTFdF = eyeC + beye + cdya_il(F, F, parallel=self.parallel)

        d2I1dFdF = 2 * cdya_ik(eye, eye, parallel=self.parallel)
        d2I2dFdF = (
            self.I1 * d2I1dFdF + 4 * dya(F, F, parallel=self.parallel) - 2 * dFFTFdF
        )

        return [
            d2WdI1dI1 * dya(self.dI1dF, self.dI1dF, parallel=self.parallel)
            + d2WdI1dI2 * dya(self.dI1dF, self.dI2dF, parallel=self.parallel)
            + d2WdI1dI2 * dya(self.dI2dF, self.dI1dF, parallel=self.parallel)
            + d2WdI2dI2 * dya(self.dI2dF, self.dI2dF, parallel=self.parallel)
            + self.dWdI1 * d2I1dFdF
            + self.dWdI2 * d2I2dFdF
        ]
