import numpy as np

from ._plotting import LabPlotter


class Simulation(LabPlotter):
    """Results of a simulation along with methods to convert and plot the data.

    Attributes
    ----------
    loadcase : class
        A class with methods for evaluating the deformation gradient and the stress
        as normal force per undeformed area, e.g. :class:`Uniaxial <.lab.Uniaxial>`.
    stretch : ndarray
        The stretch as the ratio of the deformed vs. the undeformed length.
    labels : list of str
        A list of the material parameter labels.
    material : class
        A class with a method for evaluating the gradient of the strain energy function
        w.r.t. the deformation gradient, e.g.
        :class:`DistortionalSpace <.DistortionalSpace>`.
    parameters : array_like
        The material parameters.


    Examples
    --------
    The material model response behaviour of a hyperelastic material model formulation
    is evaluated for a :class:`uniaxial tension <.lab.Uniaxial>` load case. A given
    stretch data is used to create the :class:`Simulation <.lab.Simulation>` object.

    >>> import numpy as np
    >>> import hyperelastic
    >>> from hyperelastic import lab
    >>>
    >>> stretch = np.linspace(0.7, 2.5, 181)

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

    A list of (string) labels is used to apply the list or array of parameter values to
    the material formulation.

    >>> simulation = lab.Simulation(
    >>>     loadcase=lab.Uniaxial(),
    >>>     stretch=stretch,
    >>>     material=material,
    >>>     labels=["C10", "C01", "C11", "C20", "C30"],
    >>>     parameters=[0.4, 0.1, 0.02, -0.04, 0.01],
    >>> )

    The stress-stretch plot returns a figure which visualizes the force per undeformed
    area vs. the ratio of the undeformed and deformed length.

    >>> fig, ax = simulation.plot_stress_stretch()
    >>>
    >>> ax.legend()
    >>> ax.set_title("Third-Order Deformation")

    ..  image:: images/fig_simulation-tod.png

    """

    def __init__(self, loadcase, stretch, labels, material, parameters=None):
        """Initialize the results of a simulation.

        Parameters
        ----------
        loadcase : class
            A class with methods for evaluating the deformation gradient and the stress
            as normal force per undeformed area, e.g.
            :class:`Uniaxial <.lab.Uniaxial>`.
        stretch : ndarray
            The stretch as the ratio of the deformed vs. the undeformed length.
        labels : list of str
            A list of the material parameter labels.
        material : class
            A class with a method for evaluating the gradient of the strain energy
            function w.r.t. the deformation gradient, e.g.
            :class:`DistortionalSpace <.DistortionalSpace>`.
        parameters : array_like or None, optional
            The material parameters (default is None).

        """

        super().__init__()

        self.loadcase = loadcase
        self.stretch = stretch
        self.labels = labels
        self.parameters = parameters
        self.material = material

        self.label = loadcase.label

        if self.parameters is None:
            self.parameters = np.ones(len(self.labels))

    def stress_curve_fit(self, x, *parameters):
        """Evaluate the stress as force per undeformed area for given material
        parameters."""

        kwargs = {label: p for label, p in zip(self.labels, parameters)}
        mat = self.material(**kwargs)
        F = self.loadcase.defgrad(x)
        P = mat.gradient([F.reshape(3, 3, 1, -1), None])[0].reshape(3, 3, -1)
        return self.loadcase.stress(F, P)

    def stress(self):
        "Evaluate the stress as force per undeformed area."

        return self.stress_curve_fit(self.stretch, *self.parameters)
