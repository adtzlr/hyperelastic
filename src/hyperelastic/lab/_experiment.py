from ._plotting import LabPlotter


class Experiment(LabPlotter):
    def __init__(
        self, label, displacement, force, area, length, time=None, temperature=None
    ):
        "The results of an experiment along with methods to convert and plot the data."
        self.label = label
        self.displacement = displacement
        self.force = force
        self.area = area
        self.length = length
        self.time = time
        self.temperature = temperature

        self.stretch = 1 + self.displacement / self.length

    def stress(self):
        return self.force / self.area
