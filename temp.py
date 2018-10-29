"""
This script calculates the 1h- and 2h- halo contribution of any two profiles.
"""


import numpy as np
from scipy.integrate import quad, simps
from scipy.interpolate import interp1d
import pyccl as ccl


def integ_interp(integrand, a, b, xmin, xmax,  npoints=100, *args):
    """Integrates an expression with integration bounds ``a`` and ``b``
    recursively by varying a parameter ``x`` within that expression between
    ``xmin`` and ``xmax``.

    Interpolates the resulting x versus int(expression|x) plot and returns
    an interpolating function.
    """
    x_arr = np.logspace(xmin, xmax, npoints)
    f_arr = np.array([ quad(integrand, a, b, args=args) for x in x_arr ])
    F = interp1d(x_arr, f_arr, fill_value=0)
    return F


# Cosmology Definition
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)











