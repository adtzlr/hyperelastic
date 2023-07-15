import matplotlib.pyplot as plt
import numpy as np

from ._curve_fit import concatenate_curve_fit


class Optimize:
    def __init__(self, experiments, simulations, parameters, mask=None):
        self.experiments = experiments
        self.simulations = simulations
        self.parameters = parameters
        self.mask = mask
        self.take = None

        if self.mask is None:
            self.mask = [None] * len(self.experiments)

    def init(self, *args, **kwargs):
        self.f = [simulation.stress_curve_fit for simulation in self.simulations]
        self.x = [
            experiment.stretch[mask]
            for mask, experiment in zip(self.mask, self.experiments)
        ]
        self.y = [
            experiment.stress()[mask]
            for mask, experiment in zip(self.mask, self.experiments)
        ]

        f0 = np.concatenate(
            [fi(xi, *self.parameters) for fi, xi in zip(self.f, self.x)]
        )

        self.errors = np.ones_like(self.parameters) * np.nan
        self.residuals = f0 - np.concatenate(self.y)

        for simulation in self.simulations:
            simulation.parameters = self.parameters

        return self.parameters

    def mean_relative_std(self):
        return np.mean(self.errors / np.abs(self.parameters)) * 100

    def relative_norm_residuals(self):
        return np.linalg.norm(self.residuals / np.concatenate(self.y))

    def curve_fit(self, *args, **kwargs):
        p0 = self.init(*args, **kwargs)

        popt, pcov = concatenate_curve_fit(self.f, self.x, self.y, p0, *args, **kwargs)
        fopt = np.concatenate([fi(xi, *popt) for fi, xi in zip(self.f, self.x)])

        self.parameters[:] = popt
        self.errors[:] = np.sqrt(np.diag(pcov))
        self.residuals[:] = fopt - np.concatenate(self.y)

        for simulation in self.simulations:
            simulation.parameters[:] = self.parameters

        return self.parameters, pcov

    def plot(self, title=None):
        fig, ax = plt.subplots()

        for i, (simulation, experiment) in enumerate(
            zip(self.simulations, self.experiments)
        ):
            label = experiment.label
            fig, ax = simulation.plot_stress_stretch(
                f"C{i}", label=f"{label}", lw=3, ax=ax
            )
            fig, ax = experiment.plot_stress_stretch(
                f"C{i}", label=f"{label} (Experiment)", lw=0.7, ax=ax
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
            + f"Relative Norm of Residuals = {self.relative_norm_residuals(): 1.1g}%"
            + "\n"
            + f"Mean Standard Deviation of Parameters = {self.mean_relative_std(): 1.1g}%",
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize="small",
            transform=ax.transAxes,
        )

        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        return fig, ax
