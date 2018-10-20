"""
This script contains definitions of useful cosmological functions for quick
retrival and data analysis.
"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo



h = lambda z: cosmo.H(z).value/cosmo.H0.value  # reduced Hubble's constant
rho_cr_SI = lambda z: 1e3*cosmo.critical_density(z).value  # rho_cr at z (SI)


def R_Delta(Delta, M, z):
    """
    """
    rho_cr = rho_cr_SI(z)  # rho_crit at z

    R = (3*M / (4*np.pi*Delta*rho_cr))**(1/3)
    return R
