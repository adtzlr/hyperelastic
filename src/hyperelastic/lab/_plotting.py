import matplotlib.pyplot as plt


class LabPlotter:
    "Class with methods for generating plots of experiments and simulations."

    def plot_force_displacement(
        self, *args, xlabel=r"Displacement $d$", ylabel=r"Force $F$", ax=None, **kwargs
    ):
        "Create a force-displacement plot."

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.plot(self.displacement, self.force, *args, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax

    def plot_stress_stretch(
        self,
        *args,
        ax=None,
        xlabel=r"Stretch $l/L$",
        ylabel=r"Force per undeformed area $F/A$",
        **kwargs,
    ):
        "Create a stress-stretch plot."

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ax.plot(self.stretch, self.stress(), *args, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax
