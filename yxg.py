"""
#TODO: Replace function descriptions with ones from workstation..............OK
#TODO: Check workstation imports.............................................OK

#TODO: Write up Battaglia....................................................OK
#TODO: Plot Arnaud(aP+aPP)/Arnaud(aPP) for different masses (loglog).........OK

#TODO: Class with profiles and power spectra.................................OK
#TODO: Separate variables in Arnaud..........................................OK
#TODO: NFW profile
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy.cosmology import Planck15 as cosmo
from cosmotools import h, R_Delta



class Profile():
    """This class instantiates thermal pressure profiles in haloes.
    It can compute the radial profiles and their respective Fourier transforms.
    User has to input some scaled radius, ``r/R``, where R is characterised by
    an overdensity parameter ``Delta``.

    - Note: User should input the ``M500`` mass in solar masses.
    """
    def __init__(self, x, M, z, profile=None):
        self.dict = {"Arnaud": Arnaud, "Battaglia": Battaglia}

        self.x = x
        self.M = M
        self.z = z
        self.profile = self.dict[profile]

        self.R_Delta = R_Delta(self.profile.Delta, self.M, self.z)  # FIXME: AttributeError: type object 'Arnaud' has no attribute 'Delta'
        self.fourier_interp = self.integ_interp()


    def integ_interp(self):
        """
        """
        def integrand(x, q):
            I = self.form_factor(x)*x**2*np.sinc(q*x)
            return I

        q_array = np.logspace(-3, 3, 100)
        f_array = [quad(integrand, 0, np.inf, args=q)[0] for q in q_array]
        return interp1d(q_array, np.array(f_array))


    def form_factor(self, x):
        """Computes the form factor of the profile.
        """
        return self.profile.form_factor(x)


    def fourier(self, k):  # Is k user-defined?
        """
        """
        return self.fourier_interp(k*self.R_Delta)


    def norm(self, M, z):
        """Computes the normalisation factor of the profile.
        """
        return self.profile.P0(M, z)


    def fourier_profile(self, k, M, z):
        """
        """
        F = self.norm(self.M, self.z) * self.fourier(k) * self.R_Delta**3
        return F



class Arnaud(Profile):

    def __init__(self, x, M, z):
        self.x = x
        self.M = M
        self.z = z
        self.Delta = 500  # reference overdensity (Arnaud et al.)


    def P0(self, M, z):
        """Yields the normalisation factor of the Arnaud profile.
        """
        aP = 0.12  # Arnaud et al.
        h70 = cosmo.H(0).value/0.7  # TODO: check if correct
        P0 = 8.310*h70**(-3/2)  # reference pressure

        K = 1.65e-3*h70**2*P0 / (3e14/h70)**(2/3+aP)  # prefactor
        Pz = h(self.z)**(8/3)  # redshift dependence
        PM = self.M**(2/3+aP)  # mass dependence
        P = K*Pz*PM
        return P


    def form_factor(self, x):
        """Yields the form factor of the Arnaud profile.
        """
        # Arnaud et al. best fit
        c500 = 1.156
        alpha = 1.0620
        beta = 5.4807
        gama = 0.3292

        f1 = (c500*self.x)**-gama
        f2 = (1+(c500*self.x)**alpha)**(-(beta-gama)/alpha)
        return f1*f2



class Battaglia(Profile):

    def __init__(self, x, M, z):
        self.x = x
        self.M = M
        self.z = z
        self.Delta = 200  # reference overdensity (Battaglia et al.)

    #TODO: Separate variables and write-up sub-class.
