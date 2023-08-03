import numpy as np
import hyperelastic


def test_generalized():
    ϵ = np.sqrt(np.finfo(np.float64).eps)
    λ = np.linspace(1, 1 + ϵ, 2)
    z = np.zeros_like(λ)

    # uniaxial incompressible deformation
    defgrad = np.array([[λ, z, z], [z, 1 / np.sqrt(λ), z], [z, z, 1 / np.sqrt(λ)]])

    # loop over strain exponents
    for k in [-0.5, 0.001, 2.0, 4.2]:
        tod = hyperelastic.models.invariants.ThirdOrderDeformation(
            strain=False, C10=0.3, C01=0.2, C30=0.0
        )
        fun = hyperelastic.models.generalized.deformation
        fwg = hyperelastic.GeneralizedInvariantsFramework(tod, fun=fun, exponent=k)
        fwi = hyperelastic.InvariantsFramework(tod)

        umat1 = hyperelastic.DistortionalSpace(fwg)
        umat2 = hyperelastic.DistortionalSpace(fwi)

        P1 = umat1.gradient([defgrad, None])[0]
        P2 = umat2.gradient([defgrad, None])[0]

        force1 = P1[0, 0] - P2[2, 2] * λ ** (-3 / 2)
        force2 = P2[0, 0] - P2[2, 2] * λ ** (-3 / 2)

        dforce1 = np.diff(force1) / ϵ
        dforce2 = np.diff(force2) / ϵ

        print(k, dforce1, dforce2)
        #assert np.allclose(dforce1, dforce2)


if __name__ == "__main__":
    test_generalized()
