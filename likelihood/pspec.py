import numpy as np
from scipy.integrate import simps
import pyccl as ccl



def power_spectrum(cosmo, k_arr, a, p1, p2,
                   logMrange=(6, 17), mpoints=256,
                   include_1h=True, include_2h=True, **kwargs):
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
        If True, includes the 1-halo contribution.
    include_2h : boolean
        If True, includes the 2-halo contribution.
    **kwargs : keyword arguments
        Parametrisation of the profiles.
    """
    # Profile normalisations
    Unorm = p1.profnorm(cosmo, a, **kwargs)
    Vnorm = p2.profnorm(cosmo, a, **kwargs)
    if (Unorm < 1e-16) or (Vnorm < 1e-16): return None  # deal with zero division

    # Set up integration boundaries
    logMmin, logMmax = logMrange  # log of min and max halo mass [Msun]
    mpoints = int(mpoints)        # number of integration points
    M_arr = np.logspace(logMmin, logMmax, mpoints)  # masses sampled
    Pl = ccl.linear_matter_power(cosmo, k_arr, a)   # linear matter power spectrum

    # Out-of-loop optimisations
    delta_matter = p1.Delta/ccl.omega_x(cosmo, a, "matter")  # CCL uses Delta_m
    mfunc = ccl.massfunc(cosmo, M_arr, a, delta_matter)      # mass function
    bh = ccl.halo_bias(cosmo, M_arr, a, delta_matter)        # halo bias

    # initialise integrands
    I1h, I2h_1, I2h_2 = [np.zeros((len(k_arr), len(M_arr)))  for i in range(3)]
    for m, M in enumerate(M_arr):
        U, UU = p1.fourier_profiles(cosmo, k_arr, M, a, **kwargs)
        # optimise for autocorrelation (no need to compute again)
        if p1 == p2:
            V = U; UV = UU
        else :
            V, VV = p2.fourier_profiles(cosmo, k_arr, M, a, **kwargs)
            UV = U*V

        I1h[:, m] = mfunc[m]*UV
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

    prof1_0,prof1_02 = p1.fourier_profiles(cosmo, k_arr, M_arr[0], a, **kwargs)
    if p1 == p2 :
        prof2_0 = prof1_0
        prof12_0 = prof1_02
    else :
        prof2_0,prof2_02 = p2.fourier_profiles(cosmo, k_arr, M_arr[0], a, **kwargs)
        prof12_0 = prof1_0*prof2_0

    b2h_1 += n0_2h*prof1_0
    b2h_2 += n0_2h*prof2_0
    P1h += n0_1h*prof12_0

    F = (include_1h*P1h + include_2h*(Pl*b2h_1*b2h_2)) / (Unorm*Vnorm)
    return F



def ang_power_spectrum(cosmo, l_arr, p1, p2,
                       zrange=(1e-6, 6), zpoints=128, is_zlog=True,
                       logMrange=(6, 17), mpoints=256,
                       include_1h=True, include_2h=True, **kwargs):
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
        If True, includes the 1-halo contribution.
    include_2h : boolean
        If True, includes the 2-halo contribution.
    **kwargs : keyword arguments
        Parametrisation of the profiles.
    """
    # Integration boundaries
    zmin, zmax = zrange
    # Distance measures & out-of-loop optimisations
    if is_zlog:
        z_arr = np.geomspace(zmin, zmax, zpoints)
        jac = z_arr
        x_arr= np.log(z_arr)
    else:
        z_arr = np.linspace(zmin, zmax, zpoints)
        jac = 1
        x_arr = z_arr
    a_arr = 1/(1+z_arr)
    chi_arr = ccl.comoving_radial_distance(cosmo, 1/(1+z_arr))

    c1 = ccl.h_over_h0(cosmo,a_arr)*cosmo["h"]
    invh_arr = 2997.92458 * jac/c1  # c*z/H(z)

    # Window functions
    Wu = p1.kernel(cosmo, a_arr)
    Wv = p2.kernel(cosmo, a_arr)
    N = invh_arr*Wu*Wv/chi_arr**2  # overall normalisation factor

    I = np.zeros((len(l_arr), len(chi_arr)))  # initialise integrand
    for x, chi in enumerate(chi_arr):
        k_arr = (l_arr+1/2)/chi
        Puv = power_spectrum(cosmo, k_arr, a_arr[x], p1, p2,
                             logMrange=logMrange, mpoints=mpoints,
                             include_1h=include_1h, include_2h=include_2h,
                             **kwargs)
        if Puv is None:  return None  # deal with zero division

        I[:, x] = N[x] * Puv

    Cl = simps(I, x=x_arr)
    return Cl
