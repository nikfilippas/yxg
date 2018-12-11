"""
#FIXME: kernel crashes in ccl.massfunc(cosmo, M, a) for M < 1e6

Questions:
    2. (73-78) is it a power of 10?
    3. HOD only for pressure
    4. U before HOD: 5e-15. U after HOD: 0.2.

"""


import numpy as np
from scipy.integrate import simps
from scipy.special import erf
import pyccl as ccl

import cosmotools as ct



def power_spectrum(cosmo, k_arr, a, p1, p2,
                   logMrange=(6, 17), mpoints=256,
                   include_1h=True, include_2h=True):
    """Computes the cross power spectrum of two halo profiles.

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
    p1, p2 : `profile2D._profile_` objects
        The profile isntances used in the computation.
    logMrange : tuple
        Logarithm (base-10) of the mass integration boundaries.
    mpoints : int
        Number or integration sampling points.
    include_1h : boolean
        If True, includes the 1-halo contribution
    include_2h : boolean
        If True, includes the 2-halo contribution

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
    # HOD model (Krause & Eifler, 2014)
    Mmin = 10**12.1
    M1 = 10**13.65
    M0 = 10**12.2
    sigma_lnM = 10**0.4
    alpha_sat = 1.0
    fc = 0.25

    # Set up integration boundaries
    logMmin, logMmax = logMrange  # log of min and max halo mass [Msun]
    mpoints = int(mpoints) # number of integration points
    M_arr = np.logspace(logMmin, logMmax, mpoints)  # masses sampled
    Pl = ccl.linear_matter_power(cosmo, k_arr, a)  # linear matter power spectrum

    # Correct from Delta_c to Delta_M if needed
    delta_matter = p1.Delta/ccl.omega_x(cosmo, a, "matter")

    # Out-of-loop optimisations
    mfunc = ccl.massfunc(cosmo, M_arr, a, delta_matter)  # mass function
    bh = ccl.halo_bias(cosmo, M_arr, a, delta_matter)    # halo bias

    # initialise integrands
    I1h, I2h_1, I2h_2 = [np.zeros((len(k_arr), len(M_arr)))  for i in range(3)]
    for m, M in enumerate(M_arr):
        U = p1.fourier_profile(cosmo, k_arr, M, a)
        V = p2.fourier_profile(cosmo, k_arr, M, a)

        # HOD Model
        Nc = 0.5 * (1 + erf((np.log10(M)-np.log10(Mmin))/sigma_lnM))
        Ns = np.heaviside(M-M0, 0.5) * ((M-M0)/M1)**alpha_sat
        # treat pressure profile with HOD
        if p1.is_pressure: U = ct.HOD(U, fc, Nc, Ns)
        if p2.is_pressure: V = ct.HOD(V, fc, Nc, Ns)

        I1h[:, m] = mfunc[m]*U*V
        I2h_1[:, m] = bh[m]*mfunc[m]*U
        I2h_2[:, m] = bh[m]*mfunc[m]*V

    # Tinker mass function is given in dn/dlog10M, so integrate over d(log10M)
    P1h = simps(I1h, x=np.log10(M_arr))
    b2h_1 = simps(I2h_1, x=np.log10(M_arr))
    b2h_2 = simps(I2h_2, x=np.log10(M_arr))

    # Contribution from small masses (added in the beginning)
    rhoM = ccl.rho_x(cosmo, a, "matter", is_comoving=True)
    dlM = (logMmax-logMmin) / (mpoints-1)

    n0_1h = (rhoM - np.sum(mfunc*M_arr) * dlM) / M_arr[0]
    n0_2h = (rhoM - np.sum(mfunc*bh*M_arr) * dlM) / M_arr[0]

    prof1_0 = p1.fourier_profile(cosmo, k_arr, M_arr[0], a)
    prof2_0 = p2.fourier_profile(cosmo, k_arr, M_arr[0], a)

    b2h_1 += n0_2h*prof1_0
    b2h_2 += n0_2h*prof2_0
    P1h += n0_1h*prof1_0*prof2_0

    F = include_1h*P1h + include_2h*(Pl*b2h_1*b2h_2)
    return F



def ang_power_spectrum(cosmo, l_arr, p1, p2,
                       zrange=(1e-6, 6), zpoints=128,
                       logMrange=(6, 17), mpoints=256,
                       include_1h=True, include_2h=True):
    """Computes the angular cross power spectrum of two halo profiles.

    Uses the halo model prescription for the 3D power spectrum to compute
    the angular cross power spectrum of two profiles.

    Parameters
    ----------
    cosmo : `pyccl.Cosmology` object
        Cosmological parameters.
    l_arr : array_like
        The l-values (multiple number) of the cross power spectrum.
    p1, p2 : `profile2D._profile_` objects
        The profile isntances used in the computation.
    zrange : tuple
        Minimum and maximum redshift probed.
    zpoints : int
        Number or integration sampling points in redshift.
    logMrange : tuple
        Logarithm (base-10) of the mass integration boundaries.
    mpoints : int
        Number or integration sampling points.
    include_1h : boolean
        If True, includes the 1-halo contribution
    include_2h : boolean
        If True, includes the 2-halo contribution

    Returns
    -------
    Cl : array_like
        Value of the angular power spectrum computed at each element of ``l_arr``.

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pyccl as ccl
    >>> import profile2D
    >>> cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
                              h=0.67, A_s=2.1e-9, n_s=0.96)
    >>> # plot multiple moment against Arnaud profile's autocorrelation
    >>> l_arr = np.logspace(1.2, 4, 100)  # multipole moment
    >>> p1 = profile2D.Arnaud()
    >>> p2 = profile2D.NFW()
    >>> Cl = ang_power_spectrum(cosmo, l_arr, p1, p2)
    >>> Cl_scaled = 1e12*l_arr*(l_arr+1)*Cl/(2*np.pi)
    >>> plt.loglog(l_arr, Cl_scaled)
    """
    # Integration boundaries
    zmin, zmax = zrange
    # Distance measures & out-of-loop optimisations
    z_arr = np.exp(np.linspace(np.log(zmin), np.log(zmax), zpoints))
    a_arr = 1/(1+z_arr)
    chi_arr = ccl.comoving_radial_distance(cosmo, 1/(1+z_arr))

    c1 = ccl.h_over_h0(cosmo,a_arr)*cosmo["h"]
    invh_arr = 2997.92458 * z_arr/c1 # c*z/H(z)

    # Window functions
    Wu = p1.kernel(cosmo, a_arr)
    Wv = p2.kernel(cosmo, a_arr)
    N = invh_arr*Wu*Wv/chi_arr**2  # overall normalisation factor

    I = np.zeros((len(l_arr), len(chi_arr)))  # initialise integrand
    for x, chi in enumerate(chi_arr):
        k_arr = (l_arr+1/2)/chi
        Puv = power_spectrum(cosmo, k_arr, a_arr[x], p1, p2,
                             logMrange=logMrange, mpoints=mpoints,
                             include_1h=include_1h, include_2h=include_2h)

        I[:, x] = N[x] * Puv
    Cl = simps(I, x=np.log(z_arr))
    return Cl
