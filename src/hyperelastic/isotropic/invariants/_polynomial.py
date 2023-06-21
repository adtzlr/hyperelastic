import numpy as np


class Polynomial:
    def __init__(self, C10=0, C01=0, C11=0, C20=0, C30=0):
        self.C10 = C10
        self.C01 = C01
        self.C11 = C11
        self.C20 = C20
        self.C30 = C30

        self.x = [np.eye(3), np.zeros(0)]

    def gradient(self, x):
        [I1, I2], statevars = x[:2], x[-1]

        dWdI1 = (
            self.C10
            + self.C11 * (I2 - 3)
            + self.C20 * 2 * (I1 - 3)
            + self.C30 * 3 * (I1 - 3) ** 2
        )
        dWdI2 = self.C01 + self.C11 * (I1 - 3)

        return [dWdI1, dWdI2, statevars]

    def hessian(self, x):
        I1 = x[0]

        d2WdI1I1 = self.C20 * 2 + self.C30 * 6 * (I1 - 3)
        d2WdI1I2 = self.C11
        d2WdI2I2 = 0

        return [d2WdI1I1, d2WdI1I2, d2WdI2I2]
