"""
This script contains definitions of useful cosmological functions for quick
retrieval and data analysis.
"""

from pyccl.massfunction import massfunc_m2r



def R_Delta(cosmo, halo_mass, Delta=200):
    """
    Calculate the reference radius of a halo.

    .. note:: this is R=(3M/(4*π*ρ_c))^(1/3), where ρ_c is the critical matter
              density

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
    float or array_like : The halo reference radius.
    """
    Rnorm = (cosmo["Omega_m"] / Delta)**(1/3)
    R = Rnorm * massfunc_m2r(cosmo, halo_mass)
    return R
