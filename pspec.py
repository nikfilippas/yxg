"""
"""


import numpy as np
from scipy.integrate import quad, simps
from scipy.interpolate import interp1d
import scipy.constants as u
from scipy.constants import value as v
import pyccl as ccl

from cosmotools import R_Delta



class Profile(object):
    """
    Calculate a profile quantity of a halo and its fourier transform.


    Parameters
    ----------
    profile : str
        Specifies the profile to use. Implemented profiles are 'arnaud',
        'battaglia'.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyccl as ccl
    >>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
                              h=0.67, A_s=2.1e-9, n_s=0.96)
    >>> p1 = Profile(profile="arnaud")
    >>> # radial profile is the product of the normalisation and the form factor
    >>> x = np.linspace(1e-3, 2, 100)  # R/R_Δ
    >>> radial_profile = p1.norm(cosmo, M=1e+14, a=0.7) * p1.form_factor(x)
    >>> plt.loglog(x, radial_profile)  # plot profile as a function of radius

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyccl as ccl
    >>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
                              h=0.67, A_s=2.1e-9, n_s=0.96)
    >>> p2 = Profile(profile="arnaud")
    >>> # plot the profile in fourier space
    >>> k = np.logspace(-1, 1, 100)  # wavenumber
    >>> U = p2.fourier_profile(cosmo, k, M=1e+14, a=0.6)
    >>> plt.loglog(k, U)  # plot profile in fourier space
    """
    def __init__(self, profile=None):
        # Input handling
        self.dic = {"arnaud": Arnaud(),
                    "battaglia": Battaglia()}

        try:
            self.profile = self.dic[profile.lower()]  # case-insensitive keys
        except KeyError:
            print("Profile does not exist or has not been implemented.")

        self.Delta = self.profile.Delta  # overdensity parameter
        self._fourier_interp = self._integ_interp()


    def _integ_interp(self):
        """Computes the integral of the power spectrum at different points and
        returns an interpolating function connecting these points.
        """
        def integrand(lgx):
            I = self.form_factor(lgx)*lgx
            return I

        # Integration Boundaries
        rmin, rmax = 1e-4, 1e4  # physical distance [R_Delta]
        qmin, qmax = 1/rmax, 1/rmin  # fourier space parameter
        qpoints = int(1e2)

        q_arr = np.logspace(np.log10(qmin), np.log10(qmax), qpoints)
        f_arr = [quad(integrand,
                      a=1e-4, b=np.inf,  # limits of integration
                      weight="sin", wvar=q,  # fourier sinusoidal weight
                      limit=200, limlst=100  # improve accuracy
                      )[0] / q for q in q_arr]

        F = interp1d(q_arr, np.array(f_arr), kind="cubic", fill_value=0)
        return F


    def form_factor(self, x):
        """Yields the form factor of the profile."""
        return self.profile.form_factor(x)


    def norm(self, cosmo, M, a):
        """Yields the normalisation factor of the profile."""
        return self.profile.norm(cosmo, M, a)


    def fourier_profile(self, cosmo, k, M, a):
        """Computes the Fourier transform of the full profile."""
        R = R_Delta(cosmo, M, self.Delta)  # R_Δ [Mpc]
        F = self.norm(cosmo, M, a) * self._fourier_interp(k*R) * R**3
        return F



class Arnaud(Profile):

    def __init__(self):
        self.Delta = 500  # reference overdensity (Arnaud et al.)


    def norm(self, cosmo, M, a):
        """Computes the normalisation factor of the Arnaud profile.

        .. note:: Normalisation factor is given in units of ``eV/cm^2``. \
        (Arnaud et al., 2009)
        """
        aP = 0.12  # Arnaud et al.
        h70 = cosmo["h"]/0.7
        P0 = 6.41 # reference pressure

        K = 1.65*h70**2*P0 * (h70/3e14)**(2/3+aP)  # prefactor

        Pz = ccl.h_over_h0(cosmo, a)**(8/3)  # scale factor (z) dependence
        PM = M**(2/3+aP)  # mass dependence
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



class Battaglia(Profile):

    def __init__(self):
        self.Delta = 200  # reference overdensity (Battaglia et al.)

    #TODO: Separate variables and write-up sub-class.



def power_spectrum(cosmo, k_arr, a, p1, p2):
    """Computes the cross power spectrum of two halo profiles.

    Uses the halo model prescription for the 3D power spectrum to compute
    the cross power spectrum of two profiles.

    For example, for the 1-halo term contribution,
    P1h = int(dM*dn/dM*U(k|M)*V(k|M)),
    it creates a 3-dimensional space with vectors (M, k, integrand) and
    uses `scipy.integrate.simps` to quench over the M-axis.


    Parameters
    ----------
    cosmo : `pyccl.Cosmology` object
        Cosmological parameters.
    k_arr : float or array_like
        The k-values of the cross power spectrum.
    a : float
        Scale factor.
    prof1, prof2 : `pspec.Profile` objects
        The profile isntances used in the computation.

    Returns
    -------
    f_arr : float or array_like
        Value of the cross power spectrum computed  at each element of ``k_arr``.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyccl as ccl
    >>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
                              h=0.67, A_s=2.1e-9, n_s=0.96)
    >>> # plot wavenumber against Arnaud profile's autocorrelation
    >>> k_arr = np.logspace(-1, 1, 100)  # wavenumber
    >>> P = power_spectrum(cosmo, k_arr, a=0.85, p1, p2)
    >>> plt.loglog(k_arr, P)
    """
    # Set up integration bounds
    logMmin, logMmax = 10, 16  # log of min and max halo mass [Msun]
    mpoints = int(1e2) # number of integration points
    # Tinker mass function is given in dn/dlog10M, so integrate over d(log10M)
    M_arr = np.logspace(logMmin, logMmax, mpoints)  # log10(M)
    Pl = ccl.linear_matter_power(cosmo, k_arr, a)  # linear matter power spectrum

    # initialise integrands
    I1h, I2h_1, I2h_2 = [np.zeros((len(k_arr), len(M_arr)))  for i in range(3)]
    for m, M in enumerate(M_arr):
        try:  # FIXME: fix this when done (crashes kernel if not properly input)
            U = p1.fourier_profile(cosmo, k_arr, M, a)
            V = p2.fourier_profile(cosmo, k_arr, M, a)
            mfunc = ccl.massfunc(cosmo, M, a, p1.Delta)  # mass function
            bh = ccl.halo_bias(cosmo, M, a, p1.Delta)  # halo bias

            I1h[:, m] = mfunc*U*V
            I2h_1[:, m] = bh*mfunc*U
            I2h_2[:, m] = bh*mfunc*V
        except ValueError as err:
            print(str(err)+" Try changing the range of the input wavenumber.")
            continue

    P1h = simps(I1h, x=M_arr)
    P2h = Pl*(simps(I2h_1, x=M_arr)*simps(I2h_2, x=M_arr))
    F = P1h + P2h
    return F



def ang_power_spectrum(cosmo, l_arr, a, p1, p2, zmin=1e-3, zmax=2):
    """
    """

    # Thermal Sunyaev-Zel'dovich
    sigma = v("Thomson cross section")
    prefac = sigma/(u.m_e*u.c)
    Wy = lambda a: prefac*a  # tSZ window function

    # Integration boundaries
    chimin = ccl.comoving_radial_distance(cosmo, 1/(1+zmin))
    chimax = ccl.comoving_radial_distance(cosmo, 1/(1+zmax))
    # Distance measures
    chi_arr = np.linspace(chimin, chimax, 100)
    a_arr = ccl.scale_factor_of_chi(cosmo, chi_arr)

    I = np.zeros((len(l_arr), len(chi_arr)))  # initialise integrand
    for x, chi in enumerate(chi_arr):
        k_arr = (l_arr+1/2)/chi
        Puv = power_spectrum(cosmo, k_arr, a_arr[x], p1, p2)
        W = Wy(a_arr[x])

        I[:, x] = W**2/chi * Puv

    Cl = simps(I, x=chi_arr)
    return Cl


"""
### NOTES ###
1e-4/Mpc < k < 10/Mpc
"""
