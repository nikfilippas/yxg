"""
- implemented CCL
- improved docstrings
- replaced redshift with scale factor in line with CCL
- replaced R_Delta calculations with custom function using CCL
"""


import numpy as np
from scipy.integrate import quad, simps
from scipy.interpolate import interp1d
from pyccl.massfunction import massfunc, massfunc_m2r, halo_bias
from pyccl.power import linear_matter_power as P_lin
from pyccl.background import h_over_h0 as h

from cosmotools import R_Delta



class Profile(object):
    """
    Calculate the thermal pressure profile of a halo and its fourier transform.


    Parameters
    ----------
    cosmo : (:obj: `Cosmology`) object
        Cosmological parameters.
    profile : str
        Specifies the profile to use. Implemented profiles are 'arnaud',
        'battaglia'.

    Examples
    --------
    >>> import numpy as np
    >>> import pyccl as ccl
    >>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9,
                              n_s=0.96)
    >>> p1 = Profile(cosmo, profile="arnaud")
    >>> x = np.linspace(1e-3, 2, 100)  # R/R_Δ
    >>> # radial profile is the product of the normalisation and the form factor
    >>> radial_profile = p1.norm(1e+14, 0.5) * p1.form_factor(x)

    """
    def __init__(self, cosmo, profile=None):
        # Input handling
        self.dic = {"arnaud": Arnaud(),
                    "battaglia": Battaglia()}

        try:
            self.profile = self.dic[profile.lower()]  # case-insensitive keys
        except KeyError:
            print("Profile does not exist or has not been implemented.")

        self.cosmo = cosmo
        self._fourier_interp = self._integ_interp()


    def cosmo(self):  # FIXME: maybe delete this?
        """Workaround to call the Cosmology object from subclasses."""
        return self.cosmo


    def _integ_interp(self):
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

        F = interp1d(np.log(q_arr), np.array(f_arr), kind="cubic", fill_value=0)  # TODO: check this
        return F


    def form_factor(self, x):
        """Computes the form factor of the profile.
        """
        return self.profile.form_factor(x)


    def norm(self, M, a):
        """Computes the normalisation factor of the profile.
        """
        return self.profile.P0(M, a)


    def fourier_profile(self, k, M, a):
        """Computes the Fourier transform of the full profile.
        """
        R = R_Delta(self.cosmo, M, self.profile.Delta)  # R_Δ [Mpc]
        F = self.norm(M, a) * self._fourier_interp(np.exp(k*R)) * R**3
        return F



class Arnaud(Profile):

    def __init__(self):
#        super.cosmo  # FIXME: implement this
        self.Delta = 500  # reference overdensity (Arnaud et al.)


    def P0(self, M, a):
        """Yields the normalisation factor of the Arnaud profile.
        """
        aP = 0.12  # Arnaud et al.
        h70 = super.cosmo()["h"].value/0.7
        P0 = 6.41 # reference pressure

        K = 1.65*h70**2*P0 * (h70/3e14)**(2/3+aP)  # prefactor

        Pz = h(self.cosmo, a)**(8/3)  # scale factor (redshift) dependence
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



def power_spectrum(cosmo, k_arr, a, prof1=None, prof2=None):
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
    mpoints = 1e2  # number of integration points

    M_arr = np.logspace(logMmin, logMmax, mpoints)
    I = np.zeros((len(k_arr), len(M_arr)))  # initialize
    for m, M in enumerate(M_arr):
        U = p1.fourier_profile(k_arr, M, a)
        V = p2.fourier_profile(k_arr, M, a)
        mfunc = massfunc(cosmo, M, a, p1.Delta)  # mass function

        I[:, m] = mfunc*U*V

    f_arr = simps(I, x=M_arr)
    return f_arr
