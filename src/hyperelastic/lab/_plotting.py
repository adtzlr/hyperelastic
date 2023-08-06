import matplotlib.pyplot as plt


class LabPlotter:
    "Class with methods for generating plots of experiments and simulations."

    def __init__(self):
        self.label = None

    def plot_force_displacement(
        self,
        *args,
        xlabel=r"Displacement $d$",
        ylabel=r"Force $F$",
        ax=None,
        label=None,
        **kwargs,
    ):
        "Create a force-displacement plot."

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if label is None:
            label = self.label

        ax.plot(self.displacement, self.force, *args, label=label, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax

    def plot_stress_stretch(
        self,
        *args,
        ax=None,
        xlabel=r"Stretch $l/L$",
        ylabel=r"Force per undeformed area $F/A$",
        label=None,
        **kwargs,
    ):
        "Create a stress-stretch plot."

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        if label is None:
            label = self.label

        ax.plot(self.stretch, self.stress(), *args, label=label, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax
