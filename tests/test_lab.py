import numpy as np

import hyperelastic


def pre(diameter, length):
    "Generate a synthetic experiment."
    area = diameter**2 * np.pi / 4
    stretch = np.linspace(1, 5)

    force = stretch - 1 / stretch**2 + (stretch - 1) ** 4 / 50

    stretch2 = stretch[::-1]
    stretch2 = (stretch2 - stretch2.max()) * 0.9 + stretch2.max()
    stretch = np.concatenate([stretch, stretch2])

    force2 = force * 0.5
    flow = np.linspace(1, 0, 15) ** 0.5
    force2[-15:] = flow * force2[-15:] + (1 - flow) * force[-15:]

    force = np.concatenate([force, force2[::-1]]) * area

    np.random.seed(56489)
    force += (stretch - 1) * (np.random.rand(len(stretch)) - 0.5) * 30

    displacement = (stretch - 1) * length

    return force, displacement, area, length


def material(k, **kwargs):
    tod = hyperelastic.models.invariants.ThirdOrderDeformation(strain=False, **kwargs)
    fun = hyperelastic.models.generalized.deformation
    framework = hyperelastic.GeneralizedInvariantsFramework(tod, fun=fun, exponent=k)
    return hyperelastic.DistortionalSpace(framework)


def test_lab():
    force, displacement, area, length = pre(diameter=15, length=100)

    experiments = [
        hyperelastic.lab.Experiment(
            label="Uniaxial Tension",
            displacement=displacement,
            force=force,
            area=area,
            length=length,
        ),
        hyperelastic.lab.Experiment(
            label="Biaxial Tension",
            displacement=displacement / 2,
            force=force * 2 / 3,
            area=area,
            length=length,
        ),
    ]

    ux = hyperelastic.lab.Uniaxial()
    bx = hyperelastic.lab.Biaxial()

    simulations = [
        hyperelastic.lab.Simulation(
            ux,
            experiments[0].stretch,
            material=material,
            labels=["k", "C10", "C20", "C30"],
        ),
        hyperelastic.lab.Simulation(
            bx,
            experiments[1].stretch,
            material=material,
            labels=["k", "C10", "C20", "C30"],
        ),
    ]

    optimize = hyperelastic.lab.Optimize(
        experiments=experiments,
        simulations=simulations,
        parameters=np.ones(4),
        mask=[
            np.diff(displacement, prepend=0) >= 0,  # consider only uploading path
            np.diff(displacement, prepend=0) >= 0,  # consider only uploading path
            # np.zeros_like(displacement, dtype=bool),  # deactivate biaxial loadcase
        ],
    )

    parameters, pcov = optimize.curve_fit(method="lm")
    fig, ax = optimize.plot(title="Yeoh (Generalized Invariants Framework)")

    # print(parameters)

    assert np.allclose(parameters, [1.54438747, 0.56391278, 0.01298499, 0.00233807])


if __name__ == "__main__":
    test_lab()
