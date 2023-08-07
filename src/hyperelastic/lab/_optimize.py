import matplotlib.pyplot as plt
import numpy as np

from ._curve_fit import concatenate_curve_fit


class Optimize:
    r"""Take lists of experiments and simulations and find material parameters
    for the simulation model to obtain a best possible representation of the experiments
    by the simulations.

    Attributes
    ----------
    experiments : list
        A list of :class:`Experiment <.lab.Experiment>`.
    simulations : list
        A list of :class:`Simulation <.lab.Simulation>`.
    parameters : array_like
        The material parameters.
    mask : list of array_like
        A list of masks to take the optimization-relevant data points.

    Examples
    --------
    Three different test specimens are subjected to displacement-controlled uniaxial,
    planar and biaxial tension. The applied displacement and reaction force data is
    used to create the :class:`Experiments <.lab.Experiment>`. The test specimen for the
    uniaxial load case has a cross-sectional area of :math:`A=25~\text{mm}^2` and a
    length of :math:`L=100~\text{mm}`.

    >>> area = 25
    >>> length = 100

    Some synthetic experimental data is generated to demonstrate the capabilities of the
    optimization.

    >>> import numpy as np

    >>> displacement = np.linspace(0, 2 * length, 100)
    >>> stretch = 1 + displacement / length
    >>> force = (stretch - 1 / stretch ** 2 + (stretch - 1)**5 / 10) * area

    With this reference experimental data at hand, the list of experiments is created.
    In this example, the displacement and force data as well as the cross-sectional area
    are scaled from the synthetic uniaxial experimental data to the other (synthetic)
    experiments.

    >>> from hyperelastic import lab

    >>> experiments = [
    >>>     lab.Experiment(
    >>>         label="Uniaxial Tension",
    >>>         displacement=displacement,
    >>>         force=force,
    >>>         area=area,
    >>>         length=length,
    >>>     ),
    >>>     lab.Experiment(
    >>>         label="Planar Tension",
    >>>         displacement=displacement[::2],
    >>>         force=force[::2],
    >>>         area=area / (8 / 7),
    >>>         length=length,
    >>>     ),
    >>>     lab.Experiment(
    >>>         label="Biaxial Tension",
    >>>         displacement=displacement[::2] / 2,
    >>>         force=force[::2],
    >>>         area=area / (4 / 5),
    >>>         length=length,
    >>>     ),
    >>> ]

    A function which takes the material parameters and returns the hyperelastic
    constitutive material formulation has to be provided for the simulation objects.
    Here, we use an isotropic invariant-based
    :class:`third-order deformation <.models.invariants.ThirdOrderDeformation>`
    material formulation.

    >>> def material(**kwargs):
    >>>     "A third-order deformation material formulation."
    >>>
    >>>     tod = hyperelastic.models.invariants.ThirdOrderDeformation(**kwargs)
    >>>     framework = hyperelastic.InvariantsFramework(tod)
    >>>
    >>>     return hyperelastic.DeformationSpace(framework)

    The list of labels of the material parameters is used for all simulation objects.

    >>> labels = ["C10", "C01", "C11", "C20", "C30"]

    Next, the simulations for all three loadcases are created. It is important to take
    the stretches from the according experiments.

    >>> simulations = [
    >>>     lab.Simulation(
    >>>         loadcase=lab.Uniaxial(),
    >>>         stretch=experiments[0].stretch,
    >>>         material=material,
    >>>         labels=labels,
    >>>     ),
    >>>     lab.Simulation(
    >>>         loadcase=lab.Planar(),
    >>>         stretch=experiments[1].stretch,
    >>>         material=material,
    >>>         labels=labels,
    >>>     ),
    >>>     lab.Simulation(
    >>>         loadcase=lab.Biaxial(),
    >>>         stretch=experiments[2].stretch,
    >>>         material=material,
    >>>         labels=labels,
    >>>     ),
    >>> ]

    Both the list of experiments and the list of simulations are passed to
    :class:`Optimize <.lab.Optimize>`, where its curve-fit method acts as a
    simple wrapper for :class:`scipy.optimize.curve_fit`. The initial material
    parameters are all set to one.

    >>> optimize = lab.Optimize(
    >>>     experiments=experiments,
    >>>     simulations=simulations,
    >>>     parameters=np.ones(5),
    >>> )

    >>> parameters, pcov = optimize.curve_fit(method="lm")
    >>> parameters
    array([ 0.50430357, -0.01413309,  0.0141219 , -0.01641752,  0.00492179])

    >>> fig, ax = optimize.plot(title="Third-Order Deformation")

    ..  image:: images/fig_optimize-tod.png


    """

    def __init__(self, experiments, simulations, parameters, mask=None):
        """Take lists of experiments and simulations and find material parameters
        for the simulation model to obtain a best possible representation of the
        experiments by the simulations.

        Parameters
        ----------
        experiments : list of experiments
            A list of :class:`hyperelastic.lab.Experiment`.
        simulations : list of :class:`hyperelastic.lab.Simulations`
            A list of :class:`hyperelastic.lab.Simulations`.
        parameters : array_like
            The material parameters.
        mask : list of array_like or None, optional
            A list of masks to take the optimization-relevant data points (default is
            None).

        """

        self.experiments = experiments
        self.simulations = simulations
        self.parameters = parameters
        self.mask = mask

        if self.mask is None:
            self.mask = [slice(None)] * len(self.experiments)

    def init_curve_fit(self, *args, **kwargs):
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
        """Return the relative mean of the standard deviations of the material
        parameters, normalized by the absolute mean-values of the parameters.
        """

        return np.mean(self.errors / np.abs(self.parameters)) * 100

    def norm_residuals(self):
        "Return the norm of the residuals."

        return np.linalg.norm(self.residuals)

    def curve_fit(self, *args, **kwargs):
        "Use non-linear least squares to fit a list of functions to a list of data."

        p0 = self.init_curve_fit(*args, **kwargs)

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

        ax.legend(loc=2)
        text = "\n".join(
            [
                rf"{n} = {p:1.2g} $\pm$ {2*s:1.2g}"
                for n, p, s in zip(simulation.labels, self.parameters, self.errors)
            ]
        )
        ax.text(
            0.98,
            0.02,
            text
            + "\n\n"
            + f"Absolute Norm of Residuals = {self.norm_residuals(): 1.4g}"
            + "\n"
            + f"Mean Std. Deviation of Parameters = {self.mean_relative_std(): 1.1f}%",
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize="small",
            transform=ax.transAxes,
        )

        if title is not None:
            ax.set_title(title)

        fig.tight_layout()
        return fig, ax
