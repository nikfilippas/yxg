"""
# TODO: Delta not matching in NFW and Arnaud, pyccl does not accept Delta!=200
# TODO: propagate Arnaud profile backward and forward
# TODO: NFW Fourier profile inconsistency between papers (?)
# TODO: dN/dz units?
"""

import numpy as np
from scipy.special import sici
from scipy.integrate import quad
from scipy.interpolate import interp1d
import scipy.constants as u
from scipy.constants import value as v
import pyccl as ccl

from cosmotools import R_Delta



class Arnaud(object):
    """
    Calculate an Arnaud profile quantity of a halo and its Fourier transform.


    Parameters
    ----------
    rrange : tuple
        Desired physical distance to probe (expressed in units of R_Delta).
        Change only if necessary. For distances too much outside of the
        default range the calculation might become unstable.
    qpoints : int
        Number of integration sampling points.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyccl as ccl
    >>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
                              h=0.67, A_s=2.1e-9, n_s=0.96)
    >>> p1 = Arnaud()
    >>> # radial profile is the product of the normalisation and the form factor
    >>> x = np.linspace(1e-3, 2, 100)  # R/R_Delta
    >>> radial_profile = p1.norm(cosmo, M=1e+14, a=0.7) * p1.form_factor(x)
    >>> plt.loglog(x, radial_profile)  # plot profile as a function of radius

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyccl as ccl
    >>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
                              h=0.67, A_s=2.1e-9, n_s=0.96)
    >>> p2 = Arnaud()
    >>> # plot the profile in fourier space
    >>> k = np.logspace(-1, 1, 100)  # wavenumber
    >>> U = p2.fourier_profile(cosmo, k, M=1e+14, a=0.6)
    >>> plt.loglog(k, U)  # plot profile in fourier space
    """
    def __init__(self, rrange=(1e-5, 1e6), qpoints=1e3):

        self.rrange = rrange  # range of probed distances [R_Delta]
        self.qpoints = int(qpoints)  # no of sampling points
        self.Delta = 500  # reference overdensity (Arnaud et al.)
        self.kernel = kernel.y  # associated window function

        self._fourier_interp = self._integ_interp()


    def norm(self, cosmo, M, a, b=0.4):
        """Computes the normalisation factor of the Arnaud profile.

        .. note:: Normalisation factor is given in units of ``eV/cm^3``. \
        (Arnaud et al., 2009)
        """
        aP = 0.12  # Arnaud et al.
        h70 = cosmo["h"]/0.7
        P0 = 6.41 # reference pressure

        K = 1.65*h70**2*P0 * (h70/3e14)**(2/3+aP)  # prefactor

        Pz = ccl.h_over_h0(cosmo, a)**(8/3)  # scale factor (z) dependence
        PM = (M*(1-b))**(2/3+aP)  # mass dependence
        P = K*Pz*PM
        return P


    def form_factor(self, x):
        """Computes the form factor of the Arnaud profile."""
        # Planck collaboration (2013a) best fit
        c500 = 1.81
        alpha = 1.33
        beta = 4.13
        gama = 0.31

        f1 = (c500*x)**-gama
        f2 = (1+(c500*x)**alpha)**(-(beta-gama)/alpha)
        return f1*f2


    def _integ_interp(self):
        """Computes the integral of the power spectrum at different points and
        returns an interpolating function connecting these points.
        """
        def integrand(x):
            I = self.form_factor(x)*x
            return I

        # Integration Boundaries
        rmin, rmax = self.rrange  # physical distance [R_Delta]
        qmin, qmax = 1/rmax, 1/rmin  # fourier space parameter

        q_arr = np.logspace(np.log10(qmin), np.log10(qmax), self.qpoints)
        f_arr = [quad(integrand,
                      a=1e-15, b=np.inf,  # limits of integration
                      weight="sin", wvar=q,  # fourier sinusoidal weight
                      )[0] / q for q in q_arr]

        F = interp1d(np.log10(q_arr), np.array(f_arr), kind="cubic", fill_value=0)
        return F


    def fourier_profile(self, cosmo, k, M, a, b=0.34):
        """Computes the Fourier transform of the Arnaud profile.

        .. note:: Output units are ``[norm] Mpc^3``
        """
        R = R_Delta(cosmo, M, self.Delta)  # R_Delta [Mpc]
        F = self.norm(cosmo, M, a, b) * self._fourier_interp(np.log10(k*R)) * R**3
        return F



class NFW(object):
    """Calculate a Navarro-Frenk-White profile quantity of a halo and its
    Fourier transform.
    """
    def __init__(self):
        self.kernel = kernel.g  # associated window function


    def norm(self, cosmo, M, a, Delta=200):
        """Computes the normalisation factor of the Navarro-Frenk-White profile.

        .. note:: Normalisation factor is given in units of ``M_sun/Mpc^3``.
        """
        rho = ccl.rho_x(cosmo, a, "matter")
        c = ccl.halo_concentration(cosmo, M, a, Delta)

        P = Delta/3 * rho * c**3 / (np.log(1+c)-c/(1+c))
        return P


    def form_factor(self, cosmo, x, M, a, Delta=200):
        """Computes the form factor of the Navarro-Frenk-White profile."""
        c = ccl.halo_concentration(cosmo, M, a, Delta)
        P = 1/(x*c*(1+x*c)**2)
        return P


    def fourier_profile(self, cosmo, k, M, a, Delta=200):
        """Computes the Fourier transform of the Navarro-Frenk-White profile."""
        c = ccl.halo_concentration(cosmo, M, a, Delta)
        x = k*R_Delta(cosmo, M, Delta)/c  # FIXME: drop c?

        Si1, Ci1 = sici((1+c)*x)
        Si2, Ci2 = sici(x)

        P1 = (np.log(1+c) - c/(1+c))**-1
        P2 = np.sin(x)*(Si1-Si2) + np.cos(x)*(Ci1-Ci2)
        P3 = np.sin(c*x)/((1+c)*x)

        F = P1*(P2-P3)
        return F



class kernel(object):
    """Window function definitions.

    This class contains definitions for all used window functions (kernels)
    for computation of the angular power spectrum. Multiplying the window
    function with its corresponding profile normalisation factor yields units
    of ``1/L``.
    """
    def y(cosmo, a):
        """The thermal Sunyaev-Zel'dovich anisotropy window function."""
        sigma = v("Thomson cross section")
        prefac = sigma/(u.m_e*u.c**2)
        # normalisation
        J2eV = 1/v("electron volt")
        cm2m = u.centi
        m2Mpc = 1/(u.mega*u.parsec)
        unit_norm = J2eV * cm2m**3 * m2Mpc
        return prefac*a/unit_norm


    def g(cosmo, a):
        """The galaxy number overdensity window function."""
        Hz = ccl.h_over_h0(cosmo, a)*cosmo["H0"]
        # model data
        abins = 1/(1+np.linspace(0, 1, 1001))  # scale factor bins
        data = np.loadtxt("data/2MPZ_histog_Lorentz_2.txt", skiprows=3, usecols=1)
        data = np.append(data, [0,0])  # right-most bin

        dNdz = data[np.digitize(a, abins, right=True)]
        return Hz*dNdz
