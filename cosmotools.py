"""
This script contains definitions of useful cosmological functions for quick
retrieval and data analysis.
"""


import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl



def R_Delta(cosmo, halo_mass, Delta=200):
    """
    Calculate the reference radius of a halo.

    .. note:: this is R=(3M/(4*pi*rho_c))^(1/3), where rho_c is the critical
              matter density

    Arguments
    ---------
    cosmo : ``pyccl.Cosmology`` object
        Cosmological parameters.
    halo_mass : float or array_like
        Halo mass [Msun].
    Delta : float
        Overdensity parameter.

    Returns
    -------
    float or array_like : The halo reference radius in `Mpc`.
    """
    Rnorm = (cosmo["Omega_m"] / Delta)**(1/3)
    R = Rnorm * ccl.massfunc_m2r(cosmo, halo_mass)
    return R


def dNdz():
    """Calculate the number density of halos per unit redsfhit."""
    z_arr, dNdz_arr = np.loadtxt("data/2MPZ_histog_Lorentz_2.txt",
                                 skiprows=3).T
    a_arr = 1/(1+z_arr)
    F = interp1d(a_arr, dNdz_arr, kind="cubic",
                    bounds_error=False, fill_value=0)
    return F