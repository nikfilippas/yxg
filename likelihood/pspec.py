import numpy as np
from scipy.integrate import simps
import pyccl as ccl



def power_spectrum(cosmo, k, a, profiles, logMrange=(6, 17), mpoints=128,
                   include_1h=True, include_2h=True, squeeze=True, **kwargs):
    """Computes the cross power spectrum of two halo profiles."""
    # Input handling
    a, k = np.atleast_1d(a), np.atleast_2d(k)

    # Profile normalisations
    p1, p2 = profiles
    Unorm = p1.profnorm(cosmo, a, squeeze=False, **kwargs)
    Vnorm = Unorm if p1.name == p2.name else p2.profnorm(cosmo, a, squeeze=False, **kwargs)
    if (Vnorm < 1e-16).any() or (Unorm < 1e-16).any(): return None  # zero division
    Unorm, Vnorm = Unorm[..., None], Vnorm[..., None]  # transform axes

    # Set up integration boundaries
    logMmin, logMmax = logMrange  # log of min and max halo mass [Msun]
    mpoints = int(mpoints)        # number of integration points
    M = np.logspace(logMmin, logMmax, mpoints)  # masses sampled

    # Out-of-loop optimisations
    Pl = np.array([ccl.linear_matter_power(cosmo, k[i], a) for i, a in enumerate(a)])
    Dm = p1.Delta/ccl.omega_x(cosmo, a, "matter")  # CCL uses Delta_m
    mfunc = np.array([ccl.massfunc(cosmo, M, A1, A2) for A1, A2 in zip(a, Dm)])
    bh = np.array([ccl.halo_bias(cosmo, M, A1, A2) for A1, A2 in zip(a, Dm)])
    # shape transformations
    mfunc, bh = mfunc.T[..., None], bh.T[..., None]

    U, UU = p1.fourier_profiles(cosmo, k, M, a, squeeze=False, **kwargs)
    # optimise for autocorrelation (no need to recompute)
    if p1.name == p2.name:
        V = U; UV = UU
    else :
        V, VV = p2.fourier_profiles(cosmo, k, M, a, squeeze=False, **kwargs)
        r = kwargs["r_corr"] if "r_corr" in kwargs else 0
        UV = U*V*(1+r)


    # Tinker mass function is given in dn/dlog10M, so integrate over d(log10M)
    P1h = simps(mfunc*UV, x=np.log10(M), axis=0)
    b2h_1 = simps(bh*mfunc*U, x=np.log10(M), axis=0)
    b2h_2 = simps(bh*mfunc*V, x=np.log10(M), axis=0)

    # Contribution from small masses (added in the beginning)
    rhoM = ccl.rho_x(cosmo, a, "matter", is_comoving=True)
    dlM = (logMmax-logMmin) / (mpoints-1)
    mfunc, bh = mfunc.squeeze(), bh.squeeze()  # squeeze extra dimensions

    n0_1h = np.array((rhoM - np.dot(M, mfunc) * dlM)/M[0])[None, ..., None]
    n0_2h = np.array((rhoM - np.dot(M, mfunc*bh) * dlM)/M[0])[None, ..., None]

    P1h += (n0_1h*U[0]*V[0]).squeeze()
    b2h_1 += (n0_2h*U[0]).squeeze()
    b2h_2 += (n0_2h*V[0]).squeeze()

    F = (include_1h*P1h + include_2h*(Pl*b2h_1*b2h_2)) / (Unorm*Vnorm)
    return F.squeeze() if squeeze else F



def ang_power_spectrum(cosmo, l, profiles,
                       zrange=(1e-6, 6), zpoints=32, zlog=True,
                       logMrange=(6, 17), mpoints=128,
                       include_1h=True, include_2h=True, **kwargs):
    """Computes the angular cross power spectrum of two halo profiles.

    Uses the halo model prescription for the 3D power spectrum to compute
    the angular cross power spectrum of two profiles.

    Parameters
    ----------
    cosmo : `pyccl.Cosmology` object
        Cosmological parameters.
    l : array_like
        The l-values (multiple number) of the cross power spectrum.
    profiles : tuple of `profile2D._profile_` objects
        The profile isntances used in the computation.
    zrange : tuple
        Minimum and maximum redshift probed.
    zpoints : int
        Number or integration sampling points in redshift.
    zlog : bool
        Whether to use logarithmic spacing in redshifts.
    logMrange : tuple
        Logarithm (base-10) of the mass integration boundaries.
    mpoints : int
        Number or integration sampling points.
    include_1h : bool
        If True, includes the 1-halo contribution.
    include_2h : bool
        If True, includes the 2-halo contribution.
    **kwargs : keyword arguments
        Parametrisation of the profiles.
    """
    # Integration boundaries
    zmin, zmax = zrange
    # Distance measures & out-of-loop optimisations
    if zlog:
        z = np.geomspace(zmin, zmax, zpoints)
        jac = z
        x= np.log(z)
    else:
        z = np.linspace(zmin, zmax, zpoints)
        jac = 1
        x = z
    a = 1/(1+z)
    chi = ccl.comoving_radial_distance(cosmo, a)

    H_inv = 2997.92458 * jac/(ccl.h_over_h0(cosmo, a)*cosmo["h"])  # c*z/H(z)

    # Window functions
    p1, p2 = profiles
    Wu = p1.kernel(cosmo, a)
    Wv = Wu if (p1.name == p2.name) else p2.kernel(cosmo, a)
    N = H_inv*Wu*Wv/chi**2  # overall normalisation factor

    k = (l+1/2)/chi[..., None]
    Puv = power_spectrum(cosmo, k, a, profiles, logMrange, mpoints,
                         include_1h, include_2h, squeeze=False, **kwargs)
    if type(Puv) is type(None): return None
    I = N[..., None] * Puv

    Cl = simps(I, x, axis=0)
    return Cl