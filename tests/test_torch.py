import felupe as fem
import numpy as np
import torch

import hyperelastic


def yeoh(I1, I2, I3, C10, C20, C30):
    "Yeoh isotropic hyperelastic material formulation."
    return C10 * (I1 - 3) + C20 * (I1 - 3) ** 2 + C30 * (I1 - 3) ** 3


def test_torch():
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


def run_felupe(dtype):
    torch.set_default_dtype(dtype)

    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)

    model = hyperelastic.models.invariants.TorchModel(
        yeoh, C10=0.5, C20=-0.05, C30=0.02
    )
    framework = hyperelastic.InvariantsFramework(model)
    umat = hyperelastic.DistortionalSpace(framework)
    solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)

    move = fem.math.linsteps([0, 1], num=5)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )

    job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
    job.evaluate(tol=np.sqrt(torch.finfo(dtype).eps))


def test_torch_felupe_single_precision():
    run_felupe(dtype=torch.float32)


def test_torch_felupe_double_precision():
    run_felupe(dtype=torch.float64)


if __name__ == "__main__":
    test_torch()
    test_torch_felupe_single_precision()
    test_torch_felupe_double_precision()
