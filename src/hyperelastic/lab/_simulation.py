import numpy as np

from ._plotting import LabPlotter


class Simulation(LabPlotter):
    """Results of a simulation along with methods to convert and plot the data.

    Attributes
    ----------
    loadcase : class
        A class with methods for evaluating the deformation gradient and the stress
        as normal force per undeformed area, e.g. :class:`hyperelastic.lab.Uniaxial`.
    stretch : ndarray
        The stretch as the ratio of the deformed vs. the undeformed length.
    labels : list of str
        A list of the material parameter labels.
    material : class
        A class with a method for evaluating the gradient of the strain energy function
        w.r.t. the deformation gradient, e.g. :class:`hyperelastic.DistortionalSpace`.
    parameters : array_like
        The material parameters.

    """

    def __init__(self, loadcase, stretch, labels, material, parameters=None):
        """Initialize the results of a simulation.

        Parameters
        ----------
        loadcase : class
            A class with methods for evaluating the deformation gradient and the stress
            as normal force per undeformed area, e.g.
            :class:`hyperelastic.lab.Uniaxial`.
        stretch : ndarray
            The stretch as the ratio of the deformed vs. the undeformed length.
        labels : list of str
            A list of the material parameter labels.
        material : class
            A class with a method for evaluating the gradient of the strain energy
            function w.r.t. the deformation gradient, e.g.
            :class:`hyperelastic.DistortionalSpace`.
        parameters : array_like or None, optional
            The material parameters (default is None).

        """

        self.loadcase = loadcase
        self.stretch = stretch
        self.labels = labels
        self.parameters = parameters
        self.material = material

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
