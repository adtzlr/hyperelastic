import numpy as np

import hyperelastic


def pre(diameter, length):
    "Generate a synthetic experiment."
    area = diameter**2 * np.pi / 4
    stretch = np.linspace(1, 5)

    force = stretch - 1 / stretch**2 + stretch**3.5 / 50

    stretch2 = stretch[::-1]
    stretch2 = (stretch2 - stretch2.max()) * 0.9 + stretch2.max()
    stretch = np.concatenate([stretch, stretch2])

    force2 = force * 0.5
    flow = np.linspace(1, 0, 15) ** 0.5
    force2[-15:] = flow * force2[-15:] + (1 - flow) * force[-15:]

    force = np.concatenate([force, force2[::-1]]) * area

    np.random.seed(56489)
    force += (np.random.rand(len(stretch)) - 0.5) * 100

    displacement = (stretch - 1) * length

    return force, displacement, area, length


def material(**kwargs):
    model = hyperelastic.models.invariants.ThirdOrderDeformation(**kwargs)
    framework = hyperelastic.InvariantsFramework(model)
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
            force=force * 4 / 5,
            area=area,
            length=length,
        ),
    ]

    ux = hyperelastic.lab.Uniaxial()
    bx = hyperelastic.lab.Biaxial()

    simulations = [
        hyperelastic.lab.Simulation(
            ux, experiments[0].stretch, material=material, labels=["C10", "C20", "C30"]
        ),
        hyperelastic.lab.Simulation(
            bx, experiments[1].stretch, material=material, labels=["C10", "C20", "C30"]
        ),
    ]

    optimize = hyperelastic.lab.Optimize(
        experiments=experiments,
        simulations=simulations,
        parameters=np.ones(3),
        mask=[
            np.diff(displacement, prepend=0) >= 0,  # consider only uploading path
            np.zeros_like(displacement, dtype=bool),  # deactivate biaxial loadcase
        ],
    )

    parameters = optimize.run()
    fig, ax = optimize.plot()

    assert np.allclose(parameters, [5.28491948e-01, 9.05298609e-03, 8.32226589e-05])


if __name__ == "__main__":
    test_lab()
