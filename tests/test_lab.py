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


def material(**kwargs):
    k = kwargs.pop("k")
    tod = hyperelastic.models.invariants.ThirdOrderDeformation(strain=False, **kwargs)
    fun = hyperelastic.models.generalized.deformation
    framework = hyperelastic.GeneralizedInvariantsFramework(tod, fun=fun, exponent=k)
    # framework = hyperelastic.InvariantsFramework(tod)
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
            displacement=displacement[::2] / 2,
            force=force[::2] * 2 / 3,
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
            labels=["C10", "C20", "C30", "k"],
        ),
        hyperelastic.lab.Simulation(
            bx,
            experiments[1].stretch,
            material=material,
            labels=["C10", "C20", "C30", "k"],
        ),
    ]

    optimize = hyperelastic.lab.Optimize(
        experiments=experiments,
        simulations=simulations,
        parameters=np.ones(4),
        mask=[  # consider only uploading path
            np.diff(experiments[0].displacement, prepend=0) >= 0,
            np.diff(experiments[1].displacement, prepend=0) >= 0,
            # np.zeros_like(displacement, dtype=bool),
        ],
    )

    parameters, pcov = optimize.curve_fit(method="lm")

    print(parameters)

    assert np.allclose(parameters, [0.61332567, 0.00939938, 0.00232252, 1.55141313])

    fig, ax = experiments[0].plot_force_displacement()
    fig, ax = experiments[1].plot_force_displacement(ax=ax)

    fig, ax = experiments[0].plot_stress_stretch()
    fig, ax = experiments[1].plot_stress_stretch(ax=ax)

    fig, ax = optimize.plot(title="Yeoh (Generalized Invariants Framework)")
    ax.set_xlim(None, 1.1 * ax.get_xlim()[1])


if __name__ == "__main__":
    test_lab()
