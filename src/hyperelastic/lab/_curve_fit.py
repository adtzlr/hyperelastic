from functools import wraps

import numpy as np
from scipy.optimize import curve_fit


@wraps(curve_fit)
def concatenate_curve_fit(
    f,
    xdata,
    ydata,
    *args,
    **kwargs,
):
    """Concatenate lists of functions, xdata and ydata and apply them on
    ``scipy.optimize.curve_fit()``."""

    def concatenate_f(x, *p):
        "Evaluate and return a concatenated array for a list of functions."
        return np.concatenate([fi(xi, *p) for fi, xi in zip(f, xdata)])

    return curve_fit(
        concatenate_f,
        np.concatenate(xdata),
        np.concatenate(ydata),
        *args,
        **kwargs,
    )
