import matplotlib.pyplot as plt
import numpy as np

import hyperelastic


def test_torch():
    def yeoh(I1, I2, I3, C10, C20, C30):
        "Yeoh isotropic hyperelastic material formulation."
        return C10 * (I1 - 3) + C20 * (I1 - 3) ** 2 + C30 * (I1 - 3) ** 3

    model1 = hyperelastic.models.invariants.TorchModel(
        yeoh, C10=0.5, C20=-0.05, C30=0.02
    )
    framework1 = hyperelastic.InvariantsFramework(model1)
    umat1 = hyperelastic.DistortionalSpace(framework1)

    model2 = hyperelastic.models.invariants.ThirdOrderDeformation(
        C10=0.5, C20=-0.05, C30=0.02
    )
    framework2 = hyperelastic.InvariantsFramework(model2)
    umat2 = hyperelastic.DistortionalSpace(framework2)

    # uniaxial incompressible deformation
    stretch = np.linspace(0.7, 2.5, 181)

    simulation1 = hyperelastic.lab.Simulation(
        loadcase=hyperelastic.lab.Uniaxial(),
        stretch=stretch,
        material=lambda **kwargs: umat1,
        labels=[],
        parameters=[],
    )
    x = simulation1.stress()

    simulation2 = hyperelastic.lab.Simulation(
        loadcase=hyperelastic.lab.Uniaxial(),
        stretch=stretch,
        material=lambda **kwargs: umat2,
        labels=[],
        parameters=[],
    )
    y = simulation2.stress()

    assert np.allclose(x, y)


if __name__ == "__main__":
    test_torch()
