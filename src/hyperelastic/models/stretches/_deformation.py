import numpy as np


class DeformationInvariants:
    r"""The Formulation for a Total-Lagrangian generalized-deformation invariant-based
    isotropic hyperelastic material formulation provides the material behaviour-
    independent parts for evaluating the second Piola-Kirchhoff stress tensor as well as
    its associated fourth-order elasticity tensor."""

    def __init__(self, material, strain_exponent):
        self.material = material
        self.k = strain_exponent

    def gradient(self, stretches, statevars):
        # scaled principal stretches
        self.C = C = 2 / self.k * stretches**self.k

        # strain invariants
        self.I1 = C[0] + C[1] + C[2]
        self.I2 = C[0] * C[1] + C[1] * [2] + C[2] * C[0]
        self.I3 = C[0] * C[1] * C[2]

        self.dWdI1, self.dWdI2, self.dWdI3, statevars_new = self.material.gradient(
            self.I1, self.I2, self.I3, statevars
        )

        self.dCαdλα = 2 * stretches ** (self.k - 1)

        Cβ = C[[1, 0, 0]]
        Cγ = C[[2, 2, 1]]

        self.dI1dCα = np.ones_like(C)
        self.dI2dCα = Cβ + Cγ
        self.dI3dCα = Cβ * Cγ

        dWdλα = np.zeros_like(stretches)

        if self.dWdI1 is not None:
            dWdλα += self.dWdI1 * self.dI1dCα * self.dCαdλα

        if self.dWdI2 is not None:
            dWdλα += self.dWdI2 * self.dI2dCα * self.dCαdλα

        if self.dWdI3 is not None:
            dWdλα += self.dWdI3 * self.dI3dCα * self.dCαdλα

        return dWdλα, statevars_new

    def hessian(self, stretches, statevars):
        dWdλα, statevars_new = self.gradient(stretches, statevars)
        (
            d2WdI1dI1,
            d2WdI2dI2,
            d2WdI3dI3,
            d2WdI1dI2,
            d2WdI2dI3,
            d2WdI1dI3,
        ) = self.material.hessian(self.I1, self.I2, self.I3, statevars)

        λ = stretches
        dI1dC = self.dI1dCα
        dI2dC = self.dI2dCα
        dI3dC = self.dI3dCα
        dCdλ = self.dCαdλα

        Cγ = [None, None, None, 2, 0, 1]

        d2Wdλαdλβ = np.zeros((6, *dWdλα.shape[1:]))
        idx = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]

        d2Cdλdλ = 2 * (self.k - 1) * λ ** (self.k - 2)

        for m, (α, β) in enumerate(idx):
            if d2WdI1dI1 is not None:
                d2Wdλαdλβ[m] += d2WdI1dI1 * dI1dC[α] * dCdλ[α] * dI1dC[β] * dCdλ[β]

            if d2WdI2dI2 is not None:
                d2Wdλαdλβ[m] += d2WdI2dI2 * dI2dC[α] * dCdλ[α] * dI2dC[β] * dCdλ[β]

            if d2WdI3dI3 is not None:
                d2Wdλαdλβ[m] += d2WdI3dI3 * dI3dC[α] * dCdλ[α] * dI3dC[β] * dCdλ[β]

            if d2WdI1dI2 is not None:
                d2Wdλαdλβ[m] += 2 * d2WdI1dI2 * dI1dC[α] * dCdλ[α] * dI2dC[β] * dCdλ[β]

            if d2WdI2dI3 is not None:
                d2Wdλαdλβ[m] += 2 * d2WdI2dI3 * dI2dC[α] * dCdλ[α] * dI3dC[β] * dCdλ[β]

            if d2WdI1dI3 is not None:
                d2Wdλαdλβ[m] += 2 * d2WdI1dI3 * dI1dC[α] * dCdλ[α] * dI3dC[β] * dCdλ[β]

            if α != β:
                if self.dWdI2 is not None:
                    d2Wdλαdλβ[m] += self.dWdI2 * dCdλ[α] * dCdλ[β]

                if self.dWdI3 is not None:
                    d2Wdλαdλβ[m] += self.dWdI3 * Cγ[m] * dCdλ[α] * dCdλ[β]

            if α == β:
                if self.dWdI1 is not None:
                    d2Wdλαdλβ[m] += self.dWdI1 * dI1dC[α] * d2Cdλdλ[α]

                if self.dWdI2 is not None:
                    d2Wdλαdλβ[m] += self.dWdI2 * dI2dC[α] * d2Cdλdλ[α]

                if self.dWdI3 is not None:
                    d2Wdλαdλβ[m] += self.dWdI3 * dI3dC[α] * d2Cdλdλ[α]

        return d2Wdλαdλβ
