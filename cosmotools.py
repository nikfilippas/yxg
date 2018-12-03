"""
This script contains definitions of useful cosmological functions for quick
retrieval and data analysis.
"""


import numpy as np
from scipy.interpolate import interp1d
import pyccl as ccl

def concentration_duffy(halo_mass,a,is_D500=False) :
    """
    Concentration-Mass relation from 0804.2486
    Extended to Delta=500 (critical density)

    .. note:: Returns A*(halo_mass/M_pivot)**B/a**C
              where (A,B,C) depend on the mass definition
              and M_pivot=1E12 M_sun/h

    Arguments
    ---------
    halo_mass : float or array_like
        Halo mass [Msun].
    a : float
        Scale factor
    is_D500 : boolean
        If `True`, return the extension of the original Duffy et al. relation for Delta=500.

    Returns
    -------
    float or array_like : The halo concentration.
    """
    m_pivot=2.78164E12 #Pivot mass in M_sun

    if is_D500 :
        A_Delta=3.67; B_Delta=-0.0903; C_Delta=-0.51;
    else : #2nd row in Table 1 of Duffy et al. 2008
        A_Delta=5.71; B_Delta=-0.084; C_Delta=-0.47;
    
    return A_Delta*(halo_mass/m_pivot)**B_Delta/a**C_Delta

def R_Delta(cosmo, halo_mass, a, Delta=200, is_matter=False) :
    """
    Calculate the reference radius of a halo.

    .. note:: this is R=(3M/(4*pi*rho_c(a)*Delta))^(1/3), where rho_c is the critical
              matter density

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

    Returns
    -------
    float or array_like : The halo reference radius in `Mpc`.
    """
    omega_factor=1.
    if is_matter :
        omega_factor=ccl.omega_x(cosmo,a,'matter')
    prefac=Delta*omega_factor*1.16217766E12*(cosmo['h']*ccl.h_over_h0(cosmo,a))**2
    return (halo_mass/prefac)**(1./3.)

def dNdz():
    """Calculate the number density of halos per unit redsfhit."""
    z_arr, dNdz_arr = np.loadtxt("data/2MPZ_histog_Lorentz_2.txt",
                                 skiprows=3).T
    a_arr = 1/(1+z_arr)
    F = interp1d(a_arr, dNdz_arr, kind="cubic",
                    bounds_error=False, fill_value=0)
    return F
