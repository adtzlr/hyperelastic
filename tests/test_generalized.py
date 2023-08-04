import matplotlib.pyplot as plt
import numpy as np

import hyperelastic


def test_generalized():
    ϵ = np.sqrt(np.finfo(np.float64).eps)
    λ = np.append(1, np.linspace(1, 2, 21) + ϵ)
    z = np.zeros_like(λ)

    # uniaxial incompressible deformation
    defgrad = np.array([[λ, z, z], [z, 1 / np.sqrt(λ), z], [z, z, 1 / np.sqrt(λ)]])

    # loop over strain exponents
    for k in [-0.5, 0.001, 2.0, 4.2]:
        tod = hyperelastic.models.invariants.ThirdOrderDeformation(
            strain=False, C10=0.3, C01=0.2, C20=-0.1, C30=0.05, C11=0.1
        )
        fun = hyperelastic.models.generalized.deformation
        fwg = hyperelastic.GeneralizedInvariantsFramework(tod, fun=fun, exponent=k)

        umat = hyperelastic.DeformationSpace(fwg)
        P = umat.gradient([defgrad, None])[0]

        force = P[0, 0] - P[2, 2] * 1 / (λ * np.sqrt(λ))
        dforce = np.diff(force)[0] / ϵ

        plt.plot(λ, force, label=f"k={k}")
        plt.ylim(0, 5)
        plt.legend()

        assert np.isclose(dforce, 3)


if __name__ == "__main__":
    test_generalized()
