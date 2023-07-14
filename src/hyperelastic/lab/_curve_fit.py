import numpy as np
from scipy.optimize import curve_fit


def curve_fits(f, xdata, ydata, p0, *args, **kwargs):
    """Concatenate lists of functions, xdata and ydata and apply them on
    ``scipy.optimize.curve_fit()``."""

    def fun(x, *p):
        return np.concatenate([fi(xi, *p) for fi, xi in zip(f, xdata)])

    return curve_fit(
        fun, np.concatenate(xdata), np.concatenate(ydata), p0=p0, *args, **kwargs
    )
