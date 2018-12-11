"""
This script contains definitions of useful cosmological functions for quick
retrieval and data analysis.

# TODO: return function instead of value in dNdz
"""


import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl



HOD = lambda U, fc, Nc, Ns: Nc * (fc + Ns*U)  # HOD model



def concentration_duffy(halo_mass, a, is_D500=False):
    """
    Mass-Concentration relation from 0804.2486.
    Extended to ``Delta=500`` (Delta definition uses critical density).

    .. note:: Returns ``1A*(halo_mass/M_pivot)**B/a**C``,  where (A,B,C) depend
              on the mass definition and ``M_pivot = 1e+12 M_sun/h``.

    Arguments
    ---------
    halo_mass : float or array_like
        Halo mass [Msun].
    a : float
        Scale factor
    is_D500 : boolean
        If `True`, extends of the original Duffy et al. relation to Delta=500.

    Returns
    -------
    float or array_like : The halo concentration.
    """
    m_pivot=2.78164e12  # Pivot mass [M_sun]

    if is_D500:
        A, B, C = 3.67, -0.0903, -0.51
    else: # Duffy et al. 2008 (Table 1, row 2)
        A, B, C = 5.71, -0.084, -0.47;

    return A * (halo_mass/m_pivot)**B / a**C



def R_Delta(cosmo, halo_mass, a, Delta=200, is_matter=False) :
    """
    Calculate the reference radius of a halo.

    .. note:: This is ``R = (3M/(4*pi*rho_c(a)*Delta))^(1/3)``,
              where rho_c is the critical matter density at scale factor ``a``.

    Arguments
    ---------
    cosmo : ``pyccl.Cosmology`` object
        Cosmological parameters.
    halo_mass : float or array_like
        Halo mass [Msun].
    a : float
        Scale factor
    Delta : float
        Overdensity parameter.
    is_matter : boolean
        True when R_Delta is calculated using the average matter density.
        False when R_Delta is calculated using the critical density.

    Returns
    -------
    float or array_like : The halo reference radius in `Mpc`.
    """
    if is_matter: omega_factor = ccl.omega_x(cosmo, a, "matter")
    else: omega_factor = 1

    c1 = (cosmo["h"] * ccl.h_over_h0(cosmo, a))**2
    prefac = 1.16217766e12 * Delta * omega_factor * c1

    return (halo_mass/prefac)**(1/3)



def dNdz(a):
    """Calculate the number density of halos per unit redsfhit."""
    z_arr, dNdz_arr = np.loadtxt("data/2MPZ_histog_Lorentz_2.txt",
                                 skiprows=3).T
    a_arr = 1/(1+z_arr)
    F = interp1d(a_arr, dNdz_arr, kind="cubic",
                    bounds_error=False, fill_value=0)
    return F(a)
