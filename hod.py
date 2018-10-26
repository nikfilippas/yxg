"""
This script calculates the 1h- and 2h- halo contribution of any two profiles.
"""


import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import pyccl as ccl
import yxg
from cosmotools import scale_factor



q1 = yxg.Profile("arnaud")
q2 = yxg.Profile("arnaud")

k = np.logspace(-1, 1, 100)

M = 1e14
z = 0

U = q1.fourier_profile(k, M, z)
V = q2.fourier_profile(k, M, z)




#cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)


class PowerSpectrum(object):
    """
    """
    def __init__(self, cosmo, z):
        self.a = scale_factor(z)  # scale factor at redshift z
        self.cosmo = cosmo  # cosmology object



    def integ_interp(self):
        """
        """
        def integrand_1h(logM, z, U, V):
            """Returns the integrand of the 1-halo term.
            """
            M = 10**logM  # halo mass [Msol]
            massfunc = ccl.massfunc(self.cosmo, M, self.a)

            I = massfunc*U*V
            return I


        def integrand_2h(logM, z, profile):
            """Returns *one* part of the integrand of the 2-halo term.
            """
            M = 10**logM  # halo mass [Msol]
            massfunc = ccl.massfunc(self.cosmo, M, self.a)
            bh = ccl.halo_bias(self.cosmo, M, self.a)

            I = massfunc*bh*profile
            return I

        # Boundaries of integration
        kmin, kmax = -1, 1
        npoints = 100  # number of integration points (these are interpolated)
        logMmin, logMmax = 6, 16

        k_arr = np.logspace(kmin, kmax, npoints)

        f1_arr = [quad(integrand_1h, logMmin, logMmax, args=(z,U,V)) for k in k_arr]
        f21_arr = [quad(integrand_2h, logMmin, logMmax, args=(z,U,V)) for k in k_arr]
        f22_arr = [quad(integrand_2h, logMmin, logMmax, args=(z,U,V)) for k in k_arr]

        F1 = interp1d(k_arr, np.array(f1_arr), fill_value=0)
        F21 = interp1d(k_arr, np.array(f21_arr), fill_value=0)
        F22 = interp1d(k_arr, np.array(f22_arr), fill_value=0)

        return F1, F21, F22




q = quad(integrand, 6, 16, args=(z, U, V))[0]



def __init__(self, profile=None):
    self.fourier_interp = self.integ_interp()


def integ_interp(self):
    """Computes the integral of the power spectrum at different points and
    returns an interpolating function connecting these points.
    """
    def integrand(x, q):
        I = self.form_factor(x)*x**2*np.sinc(q*x)
        return I

    q_arr = np.logspace(-3, 3, 100)
    f_arr = [quad(integrand, 0, np.inf, args=q)[0] for q in q_arr]

    F = interp1d(q_arr, np.array(f_arr), fill_value=0)
    return F









