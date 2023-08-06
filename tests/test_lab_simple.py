import numpy as np

import hyperelastic


def test_lab_simple():
    area = 25
    length = 100

    displacement = np.linspace(0, 2 * length, 100)
    stretch = 1 + displacement / length
    force = (stretch - 1 / stretch**2 + (stretch - 1) ** 5 / 10) * area

    def material(**kwargs):
        "A third-order deformation material formulation."

        tod = hyperelastic.models.invariants.ThirdOrderDeformation(**kwargs)
        framework = hyperelastic.InvariantsFramework(tod)

        return hyperelastic.DeformationSpace(framework)

    experiments = [
        hyperelastic.lab.Experiment(
            label="Uniaxial Tension",
            displacement=displacement,
            force=force,
            area=area,
            length=length,
        ),
        hyperelastic.lab.Experiment(
            label="Planar Tension",
            displacement=displacement[::2],
            force=force[::2],
            area=area / (8 / 7),
            length=length,
        ),
        hyperelastic.lab.Experiment(
            label="Biaxial Tension",
            displacement=displacement[::2] / 2,
            force=force[::2],
            area=area / (4 / 5),
            length=length,
        ),
    ]

    labels = ["C10", "C01", "C11", "C20", "C30"]
    simulations = [
        hyperelastic.lab.Simulation(
            loadcase=hyperelastic.lab.Uniaxial(),
            stretch=experiments[0].stretch,
            material=material,
            labels=labels,
        ),
        hyperelastic.lab.Simulation(
            loadcase=hyperelastic.lab.Planar(),
            stretch=experiments[1].stretch,
            material=material,
            labels=labels,
        ),
        hyperelastic.lab.Simulation(
            loadcase=hyperelastic.lab.Biaxial(),
            stretch=experiments[2].stretch,
            material=material,
            labels=labels,
        ),
    ]

    optimize = hyperelastic.lab.Optimize(
        experiments=experiments,
        simulations=simulations,
        parameters=np.ones(5),
    )

    parameters, pcov = optimize.curve_fit(method="lm")
    fig, ax = optimize.plot(title="Third-Order Deformation")

    return fig, parameters


if __name__ == "__main__":
    fig, parameters = test_lab_simple()
    print(parameters)
    fig.savefig("../docs/hyperelastic/images/fig_optimize-tod.png")
