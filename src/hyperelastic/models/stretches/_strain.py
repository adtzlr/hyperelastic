import numpy as np


class StrainInvariants:
    r"""The Formulation for a Total-Lagrangian generalized-strain invariant-based
    isotropic hyperelastic material formulation provides the material behaviour-
    independent parts for evaluating the second Piola-Kirchhoff stress tensor as well as
    its associated fourth-order elasticity tensor."""

    def __init__(self, material, strain_exponent):
        self.material = material
        self.k = strain_exponent

    def gradient(self, stretches, statevars):
        # principal strains
        self.E = E = (stretches**self.k - 1) / self.k

        # strain invariants
        self.I1 = E[0] + E[1] + E[2]
        self.I2 = E[0] * E[1] + E[1] * [2] + E[2] * E[0]
        self.I3 = E[0] * E[1] * E[2]

        self.dWdI1, self.dWdI2, self.dWdI3, statevars_new = self.material.gradient(
            self.I1, self.I2, self.I3, statevars
        )

        self.dEαdλα = stretches ** (self.k - 1)

        Eβ = E[[1, 0, 0]]
        Eγ = E[[2, 2, 1]]

        self.dI1dEα = np.ones_like(E)
        self.dI2dEα = Eβ + Eγ
        self.dI3dEα = Eβ * Eγ

        dWdλα = np.zeros_like(stretches)

        if self.dWdI1 is not None:
            dWdλα += self.dWdI1 * self.dI1dEα * self.dEαdλα

        if self.dWdI2 is not None:
            dWdλα += self.dWdI2 * self.dI2dEα * self.dEαdλα

        if self.dWdI3 is not None:
            dWdλα += self.dWdI3 * self.dI3dEα * self.dEαdλα

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
        dI1dE = self.dI1dEα
        dI2dE = self.dI2dEα
        dI3dE = self.dI3dEα
        dEdλ = self.dEαdλα

        Eγ = [None, None, None, 2, 0, 1]

        d2Wdλαdλβ = np.zeros((6, *dWdλα.shape[1:]))
        idx = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]

        d2Edλdλ = (self.k - 1) * λ ** (self.k - 2)

        for m, (α, β) in enumerate(idx):
            if d2WdI1dI1 is not None:
                d2Wdλαdλβ[m] += d2WdI1dI1 * dI1dE[α] * dEdλ[α] * dI1dE[β] * dEdλ[β]

            if d2WdI2dI2 is not None:
                d2Wdλαdλβ[m] += d2WdI2dI2 * dI2dE[α] * dEdλ[α] * dI2dE[β] * dEdλ[β]

            if d2WdI3dI3 is not None:
                d2Wdλαdλβ[m] += d2WdI3dI3 * dI3dE[α] * dEdλ[α] * dI3dE[β] * dEdλ[β]

            if d2WdI1dI2 is not None:
                d2Wdλαdλβ[m] += 2 * d2WdI1dI2 * dI1dE[α] * dEdλ[α] * dI2dE[β] * dEdλ[β]

            if d2WdI2dI3 is not None:
                d2Wdλαdλβ[m] += 2 * d2WdI2dI3 * dI2dE[α] * dEdλ[α] * dI3dE[β] * dEdλ[β]

            if d2WdI1dI3 is not None:
                d2Wdλαdλβ[m] += 2 * d2WdI1dI3 * dI1dE[α] * dEdλ[α] * dI3dE[β] * dEdλ[β]

            if α != β:
                if self.dWdI2 is not None:
                    d2Wdλαdλβ[m] += self.dWdI2 * dEdλ[α] * dEdλ[β]

                if self.dWdI3 is not None:
                    d2Wdλαdλβ[m] += self.dWdI3 * Eγ[m] * dEdλ[α] * dEdλ[β]

            if α == β:
                if self.dWdI1 is not None:
                    d2Wdλαdλβ[m] += self.dWdI1 * dI1dE[α] * d2Edλdλ[α]

                if self.dWdI2 is not None:
                    d2Wdλαdλβ[m] += self.dWdI2 * dI2dE[α] * d2Edλdλ[α]

                if self.dWdI3 is not None:
                    d2Wdλαdλβ[m] += self.dWdI3 * dI3dE[α] * d2Edλdλ[α]

        return d2Wdλαdλβ
