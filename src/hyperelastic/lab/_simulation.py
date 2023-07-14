import numpy as np

from ._plotting import LabPlotter


class Simulation(LabPlotter):
    def __init__(self, loadcase, stretch, labels, material, parameters=None):
        self.loadcase = loadcase
        self.stretch = stretch
        self.labels = labels
        self.parameters = parameters
        self.material = material

        if self.parameters is None:
            self.parameters = np.ones(len(self.labels))

    def stress_curve_fit(self, x, *parameters):
        kwargs = {label: p for label, p in zip(self.labels, parameters)}
        mat = self.material(**kwargs)
        F = self.loadcase.defgrad(x)
        P = mat.gradient([F, None])[0]
        return self.loadcase.stress(F, P)

    def stress(self):
        return self.stress_curve_fit(self.stretch, *self.parameters)
