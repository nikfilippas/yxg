"""
This script contains definitions of 3-dimensional profiles.
"""

from scipy import constants as const
from astropy.constants import M_sun, G
from astropy.cosmology import Planck15 as cosmo
from cosmotools import h, rho_cr_SI, R_Delta



def Arnaud(x, M500=3e14, z=0, aPP=True):
    """Returns the Arnaud pressure profile of a halo of a given mass,
    ``M500``, and at a given redshift, ``z``. Units are ``keV/cm^3``.

    - Note: User should input the ``M500`` mass in solar masses.
    - Note: Units of x are ``R500``.
    """
    h70 = cosmo.H(0).value/0.7

    # Parameters
    alpha_P = 0.12  # P
    alpha_PP = 0.10 - (alpha_P+0.10) * (x/0.5)**3 / (1+x/0.5)**3  # P-primed
    if not aPP: alpha_PP = 0
    P0 = 6.41  # reference pressure

    # Arnaud et al. best fit
    c500 = 1.81
    alpha = 1.33
    beta = 4.13
    gama = 0.31

    p = P0 / ((c500*x)*gama*(1+c500*x*alpha)**((beta-gama)/alpha))
    P1 = 1.65*h(z)**(8/3) * p*h70**2
    P2 = (M500/(3e14/h70))**(2/3+alpha_P+alpha_PP)
    P = P1*P2  # universal pressure profile  [eV/cm^3]

    return P


def Battaglia(x, M200=1e14, z=0):
    """Returns the Battaglia pressure profile of a halo of a given mass,
    ``M200``, and at a given redshift, ``z``. Units are ``keV/cm^3``.

    - Note: User should input the ``M200`` mass in solar masses.
    - Units of x are ``R200``.
    """
    # [J/m^3] >> [keV/cm^3]
    eV = const.value("electron volt")
    u = (1/eV)/(1e2)**3

    power_law = lambda A0, am, az: A0 * (M200/1e14)**am * (1+z)**az  # power law

    def P_Delta(M200, z):
        """Computes the self-similar amplitude for pressure.
        (Kaiser 1986; Voit 2005)
        """
        M = M200*M_sun.value  # mass in SI
        rho_cr = rho_cr_SI(z)  # rho_crit at z
        fb = cosmo.Om(z) / cosmo.Om0  # baryon fraction

        P = G.value*M*rho_cr*fb / (2*R_Delta(200, M, z))  # P_200
        return P

    # fixed parameters
    alpha = 1.0
    gama = -0.3

    # Battaglia et al. best fit parameters
    P0 = power_law(18.1, 0.154, -0.758)
    xc = power_law(0.497, -0.00865, 0.731)
    beta = power_law(4.35, 0.0393, 0.415)

    P = P0 * (x/xc)**gama * (1+(x/xc)**alpha)**-beta  # P/P_Delta
    P = P*P_Delta(M200, z)*u  # convert to [eV/cm^3]

    return P
