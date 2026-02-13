import felupe as fem
import numpy as np
import torch

import hyperelastic


def yeoh(I1, I2, I3, C10, C20, C30):
    "Yeoh isotropic hyperelastic material formulation."
    return C10 * (I1 - 3) + C20 * (I1 - 3) ** 2 + C30 * (I1 - 3) ** 3


def test_finalize():
    model = hyperelastic.models.invariants.ThirdOrderDeformation(
        C10=0.5, C20=-0.05, C30=0.02
    )
    framework = hyperelastic.InvariantsFramework(model)

    # test force- and area related configurations for stress and elasticity tensors
    for force in [0, None]:
        for area in [0, None]:
            umat = hyperelastic.DistortionalSpace(framework, force=force, area=area)
            umat = hyperelastic.DeformationSpace(
                framework,
                force=force,
                area=area,
                parallel=True,
            )

            # uniaxial incompressible deformation
            stretch = np.linspace(0.7, 2.5, 181)

            simulation = hyperelastic.lab.Simulation(
                loadcase=hyperelastic.lab.Uniaxial(),
                stretch=stretch,
                material=lambda **kwargs: umat,
                labels=[],
                parameters=[],
            )
            simulation.plot_stress_stretch()

            mesh = fem.Cube(n=3)
            region = fem.RegionHexahedron(mesh)
            field = fem.FieldContainer([fem.Field(region, dim=3)])
            boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
            solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)

            move = fem.math.linsteps([0.1], num=0)
            step = fem.Step(
                items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
            )

            job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
            job.evaluate(tol=1)


if __name__ == "__main__":
    test_finalize()
