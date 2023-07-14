import matplotlib.pyplot as plt
import numpy as np

from ._curve_fit import curve_fits


class Optimize:
    def __init__(self, experiments, simulations, parameters, mask=None):
        self.experiments = experiments
        self.simulations = simulations
        self.parameters = parameters
        self.mask = mask

    def init(self, *args, **kwargs):
        self.f = [simulation.stress_curve_fit for simulation in self.simulations]
        self.x = [experiment.stretch[self.mask] for experiment in self.experiments]
        self.y = [experiment.stress()[self.mask] for experiment in self.experiments]

        f0 = np.concatenate(
            [fi(xi, *self.parameters) for fi, xi in zip(self.f, self.x)]
        )

        self.errors = np.ones_like(self.parameters) * np.nan
        self.residuals = f0 - np.concatenate(self.y)

        for simulation in self.simulations:
            simulation.parameters = self.parameters

        return self.parameters

    def run(self, *args, **kwargs):
        p0 = self.init(*args, **kwargs)

        popt, pcov = curve_fits(self.f, self.x, self.y, p0, *args, **kwargs)
        fopt = np.concatenate([fi(xi, *popt) for fi, xi in zip(self.f, self.x)])

        self.parameters[:] = popt
        self.errors[:] = np.sqrt(np.diag(pcov))
        self.residuals[:] = fopt - np.concatenate(self.y)

        for simulation in self.simulations:
            simulation.parameters[:] = self.parameters

        return self.parameters

    def plot(self):
        fig, ax = plt.subplots()

        for simulation, experiment in zip(self.simulations, self.experiments):
            fig, ax = simulation.plot_stress_stretch(
                "C7", label=f"{experiment.label} (Simulation)", ax=ax
            )
            fig, ax = experiment.plot_stress_stretch(
                ".-", label=f"{experiment.label} (Experiment)", ax=ax
            )

        ax.legend()
        text = "\n".join(
            [
                rf"{n} = {p:1.2g} $\pm${2*s:1.2g}"
                for n, p, s in zip(simulation.labels, self.parameters, self.errors)
            ]
        )
        ax.text(
            0.98,
            0.02,
            text
            + "\n\n"
            + f"Norm of Residuals = {np.linalg.norm(self.residuals): 1.2g}"
            + "\n"
            + f"Mean Standard Deviation of Parameters = {np.mean(self.errors): 1.2g}",
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize="small",
            transform=ax.transAxes,
        )
        fig.tight_layout()
        return fig, ax
