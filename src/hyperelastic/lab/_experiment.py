from ._plotting import LabPlotter


class Experiment(LabPlotter):
    """Results of an experiment along with methods to convert and plot the data.

    Attributes
    ----------
    label : str
        The title of the experiment.
    displacement : array_like
        The measured or applied displacement data.
    force : array_like
        The measured or applied force data.
    area : float
        The undeformed reference cross-sectional area used to evaluate the stress.
    length : float
        The undeformed reference length used to evaluate the stretch.
    time : array_like or None
        The timetrack of the measurement.
    temperature : array_like or None
        The measured or applied temperature data.
    stretch : array_like
        The stretch as the calculated ratio of the deformed vs. the undeformed length.

    """

    def __init__(
        self,
        label,
        displacement,
        force,
        area=1.0,
        length=1.0,
        time=None,
        temperature=None,
    ):
        """Initialize the results of an experiment.

        Parameters
        ----------
        label : str
            The title of the experiment.
        displacement : array_like
            The measured or applied displacement data.
        force : array_like
            The measured or applied force data.
        area : float, optional
            The undeformed reference cross-sectional area used to evaluate the stress
            (default is 1.0).
        length : float, optional
            The undeformed reference length used to evaluate the stretch (default is
            1.0).
        time : array_like or None, optional
            The timetrack of the measurement (default is None).
        temperature : array_like or None, optional
            The measured or applied temperature data (default is None).

        """

        super().__init__()

        self.label = label
        self.displacement = displacement
        self.force = force
        self.area = area
        self.length = length
        self.time = time
        self.temperature = temperature

        self.stretch = 1 + self.displacement / self.length

    def stress(self):
        "Evaluate the stress as force per undeformed area."

        return self.force / self.area
