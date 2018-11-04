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
    rrange : tuple
        Desired physical distance to probe (expressed in units of R_Δ).
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
    def __init__(self, profile=None, rrange=(1e-4, 1e4), qpoints=1e2):
        # Input handling
        self.dic = {"arnaud": Arnaud(),
                    "battaglia": Battaglia()}

        try:
            self.profile = self.dic[profile.lower()]  # case-insensitive keys
        except KeyError:
            print("Profile does not exist or has not been implemented.")

        self.rrange = rrange  # range of probed distances [R_Δ]
        self.qpoints = int(qpoints)  # no of sampling points
        self.Delta = self.profile.Delta  # overdensity parameter

        self._fourier_interp = self._integ_interp()


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
                      a=1e-4, b=np.inf,  # limits of integration
                      weight="sin", wvar=q,  # fourier sinusoidal weight
                      limit=200, limlst=100  # improve accuracy
                      )[0] / q for q in q_arr]

        F = interp1d(np.log10(q_arr), np.array(f_arr), kind="cubic", fill_value=0)
        return F


    def form_factor(self, x):
        """Yields the form factor of the profile."""
        return self.profile.form_factor(x)


    def norm(self, cosmo, M, a):
        """Yields the normalisation factor of the profile."""
        return self.profile.norm(cosmo, M, a)


    def fourier_profile(self, cosmo, k, M, a):
        """Computes the Fourier transform of the full profile.

        .. note:: Output units are ``[norm]*Mpc^3``
        """
        R = R_Delta(cosmo, M, self.Delta)  # R_Δ [Mpc]
        F = self.norm(cosmo, M, a) * self._fourier_interp(np.log10(k*R)) * R**3
        return F



class Arnaud(Profile):

    def __init__(self):
        self.Delta = 500  # reference overdensity (Arnaud et al.)


    def norm(self, cosmo, M, a):
        """Computes the normalisation factor of the Arnaud profile.

        .. note:: Normalisation factor is given in units of ``eV/cm^3``. \
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



def power_spectrum(cosmo, k_arr, a, p1, p2,
                   logMrange=(10, 16), mpoints=100, full_output=True):
    """Computes the linear cross power spectrum of two halo profiles.

    Uses the halo model prescription for the 3D power spectrum to compute
    the linear cross power spectrum of two profiles.

    For example, for the 1-halo term contribution, and for fourier space
    profiles U(k) and V(k),
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
    p1, p2 : `pspec.Profile` objects
        The profile isntances used in the computation.
    logMrange : tuple
        Logarithm (base-10) of the mass integration boundaries.
    mpoints : int
        Number or integration sampling points.

    Returns
    -------
    f_arr : float or array_like
        Value of the cross power spectrum computed  at each element of ``k_arr``.

    .. note:: Output units are ``([norm]^2 Mpc^3)``.

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
    # Set up integration boundaries
    logMmin, logMmax = logMrange  # log of min and max halo mass [Msun]
    mpoints = int(mpoints) # number of integration points
    # Tinker mass function is given in dn/dlog10M, so integrate over d(log10M)
    M_arr = np.logspace(logMmin, logMmax, mpoints)  # log10(M)
    Pl = ccl.linear_matter_power(cosmo, k_arr, a)  # linear matter power spectrum

    # initialise integrands
    I1h, I2h_1, I2h_2 = [np.zeros((len(k_arr), len(M_arr)))  for i in range(3)]
    for m, M in enumerate(M_arr):
        try:
            U = p1.fourier_profile(cosmo, k_arr, M, a)
            V = p2.fourier_profile(cosmo, k_arr, M, a)
            mfunc = ccl.massfunc(cosmo, M, a, p1.Delta)  # mass function
            bh = ccl.halo_bias(cosmo, M, a, p1.Delta)  # halo bias

            I1h[:, m] = mfunc*U*V
            I2h_1[:, m] = bh*mfunc*U
            I2h_2[:, m] = bh*mfunc*V
        except ValueError as err:
            msg = str(err)+"\nTry changing the range of the input wavenumber."
            if full_output: print(msg)
            continue

    P1h = simps(I1h, x=M_arr)
    P2h = Pl*(simps(I2h_1, x=M_arr)*simps(I2h_2, x=M_arr))
    F = P1h + P2h
    return F



def ang_power_spectrum(cosmo, l_arr, p1, p2, W1, W2,
                       zrange=(1e-3,2), chipoints=500):
    """Computes the angular cross power spectrum of two halo profiles.

    Uses the halo model prescription for the 3D power spectrum to compute
    the angular cross power spectrum of two profiles.

    Parameters
    ----------
    cosmo : `pyccl.Cosmology` object
        Cosmological parameters.
    l_arr : float or array_like
        The l-values (multiple number) of the cross power spectrum.
    p1, p2 : `pspec.Profile` objects
        The profile isntances used in the computation.
    W1, W2 : `psepc.kernel.window_function` method
        The correspoding window function kernels for the profiles.
    zrange : tuple
        Minimum and maximum redshift probed.
    chipoints : int
        Number or integration sampling points.

    Returns
    -------
    Cl : float or array_like
        Value of the angular power spectrum computed at each element of ``l_arr``.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyccl as ccl
    >>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
                              h=0.67, A_s=2.1e-9, n_s=0.96)
    >>> # plot multiple moment against Arnaud profile's autocorrelation
    >>> l_arr = np.logspace(1, 4, 100)  # multipole moment
    >>> Cl = ang_power_spectrum(cosmo, l_arr, p1, p2, kernel.tSZ, kernel.tSZ)
    >>> plt.loglog(l_arr, Cl)
    >>> Cl_scaled = 1e12*l_arr*(l_arr+1)*Cl/(2*np.pi)
    >>> plt.loglog(l_arr, Cl_scaled)
    """
    # Integration boundaries
    zmin, zmax = zrange
    chimin = ccl.comoving_radial_distance(cosmo, 1/(1+zmin))
    chimax = ccl.comoving_radial_distance(cosmo, 1/(1+zmax))
    # Distance measures
    chi_arr = np.linspace(chimin, chimax, int(chipoints))
    a_arr = ccl.scale_factor_of_chi(cosmo, chi_arr)

    I = np.zeros((len(l_arr), len(chi_arr)))  # initialise integrand
    for x, chi in enumerate(chi_arr):
        k_arr = (l_arr+1/2)/chi
        Puv = power_spectrum(cosmo, k_arr, a_arr[x], p1, p2)
        Wu = W1(a_arr[x])
        Wv = W2(a_arr[x])

        I[:, x] = Wu*Wv/chi**2 * Puv

    Cl = simps(I, x=chi_arr)
    return Cl



class kernel(object):

    def tSZ(a):
        sigma = v("Thomson cross section")
        prefac = sigma/(u.m_e*u.c)
        # normalisation
        J_to_eV = 1/v("electron volt")
        cm3_to_m3 = (u.centi)**3
        m_to_Mpc = 1/(u.mega*u.parsec)
        unit_norm = J_to_eV * cm3_to_m3 * m_to_Mpc
        return prefac*a*unit_norm
