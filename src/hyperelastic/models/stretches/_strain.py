import numpy as np


class GeneralizedStrainInvariants:
    "Generalized Strain Invariants."

    def __init__(self, material, k=2):
        self.material = material
        self.k = k

    def gradient(self, stretches, statevars):
        # principal strains
        E = (stretches**self.k - 1) / self.k

        # strain invariants
        self.I1 = E[0] + E[1] + E[2]
        self.I2 = E[0] * E[1] + E[1] * [2] + E[2] * E[0]
        self.I3 = E[0] * E[1] * E[2]

        self.dWdI1, self.dWdI2, self.dWdI3, statevars_new = self.material.gradient(
            self.I1, self.I2, self.I3, statevars
        )

        dEαdλα = stretches ** (self.k - 1)

        Eβ = E[[1, 2, 0]]
        Eγ = E[[2, 0, 1]]

        self.dI1dλα = dEαdλα
        self.dI2dλα = dEαdλα * (Eβ + Eγ)
        self.dI3dλα = dEαdλα * Eβ * Eγ

        dWdλα = (
            self.dWdI1 * self.dI1dλα
            + self.dWdI2 * self.dI2dλα
            + self.dWdI3 * self.dI3dλα
        )

        return dWdλα, statevars_new

    def hessian(self, stretches, statevars):
        return
        # d2Wdλαdλα = np.zeros_like(stretches)

        # dWdλα, statevars = self.gradient(stretches, statevars)
        # (
        #     d2WdI1dI1,
        #     d2WdI2dI2,
        #     d2WdI3dI3,
        #     d2WdI1dI2,
        #     d2WdI2dI3,
        #     d2WdI1dI3,
        # ) = self.material.hessian(self.I1, self.I2, self.I3, statevars)

        # out = np.zeros((6, *dWdλα[1:]))
        # out[:3] += self.dWdI1 * (self.k - 1) * stretches ** (self.k - 2)

        # for m, (α, β) in enumerate(np.triu_indices(3)):
        #     out[m] = (
        #         d2WdI1dI1 * self.dI1dλα[α] * self.dI1dλα[β]
        #         + d2WdI2dI2 * self.dI2dλα[α] * self.dI2dλα[β]
        #         + d2WdI3dI3 * self.dI3dλα[α] * self.dI3dλα[β]
        #         + d2WdI1dI2 * self.dI1dλα[α] * self.dI2dλα[β]
        #         + d2WdI2dI3 * self.dI2dλα[α] * self.dI3dλα[β]
        #         + d2WdI1dI3 * self.dI1dλα[α] * self.dI3dλα[β]
        #     )

        # dI1dλα_dI1dλβ = np.expand_dims(self.dI1dλα, 1) * np.expand_dims(self.dI1dλα, 0)
        # dI2dλα_dI2dλβ = np.expand_dims(self.dI2dλα, 1) * np.expand_dims(self.dI2dλα, 0)
        # dI3dλα_dI3dλβ = np.expand_dims(self.dI3dλα, 1) * np.expand_dims(self.dI3dλα, 0)
        # dI1dλα_dI2dλβ = np.expand_dims(self.dI1dλα, 1) * np.expand_dims(self.dI2dλα, 0)
        # dI2dλα_dI3dλβ = np.expand_dims(self.dI2dλα, 1) * np.expand_dims(self.dI3dλα, 0)
        # dI1dλα_dI3dλβ = np.expand_dims(self.dI1dλα, 1) * np.expand_dims(self.dI3dλα, 0)

        # d2I1dλαdλβ = (self.k - 1) * stretches ** (self.k - 2)

        # work in progress...

        # d2I2dλαdλβ =
        # d2I3dλαdλβ =

        # out = (
        #     d2WdI1dI1 * dI1dλα_dI1dλβ +
        #     d2WdI2dI2 * dI2dλα_dI2dλβ +
        #     d2WdI3dI3 * dI3dλα_dI3dλβ +
        #     d2WdI1dI2 * dI1dλα_dI2dλβ +
        #     d2WdI2dI3 * dI2dλα_dI3dλβ +
        #     d2WdI1dI3 * dI1dλα_dI3dλβ +
        #     self.dWdI1 * d2I1dλαdλβ + self.dWdI2 * d2I2dλαdλβ + self.dWdI3 * d2I3dλαdλβ
        # )

        # idx = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
        # return [out[idx]]
