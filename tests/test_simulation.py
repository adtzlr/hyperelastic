import numpy as np

import hyperelastic
from hyperelastic import lab


def test_simulation():
    stretch = np.linspace(0.7, 2.5, 181)

    def material(**kwargs):
        "A third-order deformation material formulation."

        tod = hyperelastic.models.invariants.ThirdOrderDeformation(**kwargs)
        framework = hyperelastic.InvariantsFramework(tod)

        return hyperelastic.DeformationSpace(framework)

    simulation = lab.Simulation(
        loadcase=lab.Uniaxial(),
        stretch=stretch,
        material=material,
        labels=["C10", "C01", "C11", "C20", "C30"],
        parameters=[0.4, 0.1, 0.02, -0.04, 0.01],
    )

    fig, ax = simulation.plot_stress_stretch()
    ax.legend()
    ax.set_title("Third-Order Deformation")

    return fig


if __name__ == "__main__":
    fig = test_simulation()
    fig.savefig("../docs/hyperelastic/images/fig_simulation-tod.png")
