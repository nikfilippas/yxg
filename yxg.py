"""
#TODO: NFW profile
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as cosmo
from cosmotools import h, R_Delta



class Profile(object):
    """This class instantiates thermal pressure profiles in haloes.
    It can compute the radial profiles and their respective Fourier transforms.
    User has to input some scaled radius, ``r/R``, where ``R`` is characterised
    by an overdensity parameter ``Delta``.

    Each profile has a different default value of ``Delta``, so the user simply
    has to create a ``Profile`` object with argument a string containing the
    profile of choice.

    - Note: User should input the ``M500`` mass in solar masses.
    """
    def __init__(self, profile=None):
        self.dict = {"arnaud": Arnaud(), "battaglia": Battaglia()}

        try:
            self.profile = self.dict[profile.lower()]  # case-insensitive keys
        except KeyError:
            print("Profile does not exist or has not been implemented.")

        self.fourier_interp = self.integ_interp()


    def integ_interp(self):
        """Computes the integral of the power spectrum at different points and
        returns an interpolating function connecting these points.
        """
        def integrand(x, q):
            I = self.form_factor(x)*x**2*np.sinc(q*x)
            return I

        q_array = np.logspace(-3, 3, 1000)
        f_array = [quad(integrand, 0, np.inf, args=q)[0] for q in q_array]

        F = interp1d(q_array, np.array(f_array), fill_value=0)
        return F


    def form_factor(self, x):
        """Computes the form factor of the profile.
        """
        return self.profile.form_factor(x)


    def norm(self, M, z):
        """Computes the normalisation factor of the profile.
        """
        return self.profile.P0(M, z)


    def fourier_profile(self, k, M, z):
        """Computes the Fourier transform of the full profile.
        """
        R = R_Delta(self.profile.Delta, M, z)
        F = self.norm(M, z) * self.fourier_interp(k*R) * self.R**3
        return F



class Arnaud(Profile):

    def __init__(self):
        self.Delta = 500  # reference overdensity (Arnaud et al.)


    def P0(self, M, z):
        """Yields the normalisation factor of the Arnaud profile.
        """
        aP = 0.12  # Arnaud et al.
        h70 = cosmo.H(0).value/0.7
        P0 = 6.41 # reference pressure

        K = 1.65*h70**2*P0 / (3e14/h70)**(2/3+aP)  # prefactor
        Pz = h(z)**(8/3)  # redshift dependence
        PM = M**(2/3+aP)  # mass dependence
        P = K*Pz*PM
        return P


    def form_factor(self, x):
        """Yields the form factor of the Arnaud profile.
        """
        # Planck collaboration (2013a) best fit
        c500 = 1.81
        alpha = 1.33
        beta = 4.13
        gama = 0.31

        f1 = (c500*x)**-gama
        f2 = (1+(c500*x)**alpha)**(-(beta-gama)/alpha)
        return f1*f2



class Battaglia(Profile):

    def __init__(self):
        self.Delta = 200  # reference overdensity (Battaglia et al.)

    #TODO: Separate variables and write-up sub-class.
