"""
# FIXME: form factor right, now many orders of magnitude off
"""


import numpy as np
from scipy.integrate import simps
import pyccl as ccl



def power_spectrum(cosmo, k_arr, a, p1, p2,
                   logMrange=(10, 17), mpoints=100, full_output=True):
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
    k_arr : array_like
        The k-values of the cross power spectrum.
    a : float
        Scale factor.
    p1, p2 : `profile._profile_` objects
        The profile isntances used in the computation.
    logMrange : tuple
        Logarithm (base-10) of the mass integration boundaries.
    mpoints : int
        Number or integration sampling points.

    Returns
    -------
    f_arr : array_like
        Value of the cross power spectrum computed  at each element of ``k_arr``.

    .. note:: Output units are ``([p1.norm]*[p2.norm] Mpc^3)``.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyccl as ccl
    >>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
                              h=0.67, A_s=2.1e-9, n_s=0.96)
    >>> # plot wavenumber against Arnaud profile's autocorrelation
    >>> k_arr = np.logspace(-1, 1, 100)  # wavenumber
    >>> P = power_spectrum(cosmo, k_arr, 0.85, p1, p2)
    >>> plt.loglog(k_arr, P)
    """
    # Set up integration boundaries
    logMmin, logMmax = logMrange  # log of min and max halo mass [Msun]
    mpoints = int(mpoints) # number of integration points
    M_arr = np.logspace(logMmin, logMmax, mpoints)  # masses sampled
    Pl = ccl.linear_matter_power(cosmo, k_arr, a)  # linear matter power spectrum
    # Out-of-loop optimisations
    mfunc = ccl.massfunc(cosmo, M_arr, a, p1.Delta)  # mass function
    bh = ccl.halo_bias(cosmo, M_arr, a, p1.Delta)  # halo bias

    # initialise integrands
    I1h, I2h_1, I2h_2 = [np.zeros((len(k_arr), len(M_arr)))  for i in range(3)]
    for m, M in enumerate(M_arr):
        try:
            U = p1.fourier_profile(cosmo, k_arr, M, a)
            V = p2.fourier_profile(cosmo, k_arr, M, a)


            I1h[:, m] = mfunc[m]*U*V
            I2h_1[:, m] = bh[m]*mfunc[m]*U
            I2h_2[:, m] = bh[m]*mfunc[m]*V
        except ValueError as err:
            msg = str(err)+"\nTry changing the range of the input wavenumber."
            if full_output: print(msg)
            continue
    # Tinker mass function is given in dn/dlog10M, so integrate over d(log10M)
    P1h = simps(I1h, x=np.log10(M_arr))
    P2h = Pl*(simps(I2h_1, x=np.log10(M_arr))*simps(I2h_2, x=np.log10(M_arr)))
    F = P1h + P2h
    return F



def ang_power_spectrum(cosmo, l_arr, p1, p2, W1, W2,
                       zrange=(1e-3,6), chipoints=500):
    """Computes the angular cross power spectrum of two halo profiles.

    Uses the halo model prescription for the 3D power spectrum to compute
    the angular cross power spectrum of two profiles.

    Parameters
    ----------
    cosmo : `pyccl.Cosmology` object
        Cosmological parameters.
    l_arr : array_like
        The l-values (multiple number) of the cross power spectrum.
    p1, p2 : `pspec.pressure` objects
        The profile isntances used in the computation.
    W1, W2 : `psepc.kernel.window_function` method
        The correspoding window function kernels for the profiles.
    zrange : tuple
        Minimum and maximum redshift probed.
    chipoints : int
        Number or integration sampling points.

    Returns
    -------
    Cl : array_like
        Value of the angular power spectrum computed at each element of ``l_arr``.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyccl as ccl
    >>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
                              h=0.67, A_s=2.1e-9, n_s=0.96)
    >>> # plot multiple moment against Arnaud profile's autocorrelation
    >>> l_arr = np.logspace(1.2, 4, 100)  # multipole moment
    >>> Cl = ang_power_spectrum(cosmo, l_arr, p1, p2, kernel.y, kernel.y)
    >>> Cl_scaled = 1e12*l_arr*(l_arr+1)*Cl/(2*np.pi)
    >>> plt.loglog(l_arr, Cl_scaled)
    """
    # Integration boundaries
    zmin, zmax = zrange
    chimin = ccl.comoving_radial_distance(cosmo, 1/(1+zmin))
    chimax = ccl.comoving_radial_distance(cosmo, 1/(1+zmax))
    # Distance measures & out-of-loop optimisations
    chi_arr = np.linspace(chimin, chimax, int(chipoints))
    a_arr = ccl.scale_factor_of_chi(cosmo, chi_arr)
    # Window functions
    Wu = W1(cosmo, a_arr)
    Wv = W2(cosmo, a_arr)
    N = Wu*Wv/chi_arr**2  # overall normalisation factor

    I = np.zeros((len(l_arr), len(chi_arr)))  # initialise integrand
    for x, chi in enumerate(chi_arr):
        k_arr = (l_arr+1/2)/chi
        Puv = power_spectrum(cosmo, k_arr, a_arr[x], p1, p2)

        I[:, x] = N[x] * Puv

    Cl = simps(I, x=chi_arr)
    return Cl
