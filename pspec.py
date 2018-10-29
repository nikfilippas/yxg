"""
"""


import numpy as np
from scipy.integrate import quad, simps
from scipy.interpolate import interp1d

from cosmotools import h, R_Delta, scale_factor  # TODO: replace with CCL



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
    def __init__(self, cosmo, profile=None):
        # Input handling
        self.dic = {"arnaud": Arnaud(),
                    "battaglia": Battaglia()}

        try:
            self.profile = self.dic[profile.lower()]  # case-insensitive keys
        except KeyError:
            print("Profile does not exist or has not been implemented.")

        self.cosmo = cosmo  # TODO: implement in subclass
        self.fourier_interp = self.integ_interp()


    def integ_interp(self):
        """Computes the integral of the power spectrum at different points and
        returns an interpolating function connecting these points.
        """
        def integrand(x, q):
            I = self.form_factor(x)*x**2*np.sinc(q*x)
            return I

        # Integration Boundaries
        rmin, rmax = 1e-4, 1e3  # physical distance [R_Delta]
        qmin, qmax = 1/rmax, 1/rmin  # fourier space distance [1/R_Delta]
        qpoints = 1e2

        q_arr = np.logspace(np.log10(qmin), np.log10(qmax), qpoints)
        f_arr = [quad(integrand, 0, np.inf, args=q, limit=100)[0] for q in q_arr]

        F = interp1d(np.log(q_arr), np.array(f_arr), kind="cubic", fill_value=0)
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
        F = self.norm(M, z) * self.fourier_interp(np.exp(k*R)) * R**3
        return F



class Arnaud(Profile):

    def __init__(self):
        self.Delta = 500  # reference overdensity (Arnaud et al.)


    def P0(self, M, z):
        """Yields the normalisation factor of the Arnaud profile.
        """
        aP = 0.12  # Arnaud et al.
#        h70 = self.cosmo["h"].value/0.7
        h70 = h(0)/0.7  # FIXME: replace with CCL (see above)
        P0 = 6.41 # reference pressure

        K = 1.65*h70**2*P0 * (h70/3e14)**(2/3+aP)  # prefactor
        Pz = h(z)**(8/3)  # redshift dependence  # FIXME: replace with CCL
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



def power_spectrum(cosmo, k_arr, z, prof1=None, prof2=None):
    """Uses the halo model prescription for the 3D power spectrum to compute
    the cross power spectrum of two profiles, ``prof1`` and ``prof2``.

    - User has to input the names of the profiles as strings, and the algorithm
    will take care of the rest.
    """
    # Set up Profile object
    p1 = Profile(cosmo, prof1)
    p2 = Profile(cosmo, prof2)

    # Set up integration bounds
    logMmin, logMmax = 10, 16  # log of min and max halo mass [Msol]
    mpoints = 10  # number of integration points

    M_arr = np.logspace(logMmin, logMmax, mpoints)
    I = np.zeros((len(k_arr), len(M_arr)))  # initialize
    for m, M in enumerate(M_arr):
        U = p1.fourier_profile(k_arr, M, z)
        V = p2.fourier_profile(k_arr, M, z)
#        mfunc = ccl.massfunc(cosmo, M, scale_factor(z), p1.Delta)
        mfunc = 1e-4  # FIXME: replace with CCL (see above)

        I[:, m] = mfunc*U*V

    f_arr = simps(I, x=M_arr)
    return f_arr
