"""
This script contains definitions of useful cosmological functions for quick
retrieval and data analysis.
"""

import numpy as np
from scipy.integrate import simps
import pyccl as ccl



def concentration_duffy(halo_mass, a, is_D500=False):
    """
    Mass-Concentration relation from 0804.2486.
    Extended to ``Delta=500`` (Delta definition uses critical density).

    .. note:: Returns ``1A*(halo_mass/M_pivot)**B/a**C``,  where (A,B,C) depend
              on the mass definition and ``M_pivot = 1e+12 M_sun/h``.

    Arguments
    ---------
    halo_mass : float or array_like
        Halo mass [Msun].
    a : float
        Scale factor
    is_D500 : boolean
        If `True`, extends of the original Duffy et al. relation to Delta=500.

    Returns
    -------
    float or array_like : The halo concentration.
    """
    m_pivot=2.78164e12  # Pivot mass [M_sun]

    if is_D500:
        A, B, C = 3.67, -0.0903, -0.51
    else: # Duffy et al. 2008 (Table 1, row 2)
        A, B, C = 5.71, -0.084, -0.47;

    return A * (halo_mass/m_pivot)**B / a**C



def R_Delta(cosmo, halo_mass, a, Delta=200, is_matter=False) :
    """
    Calculate the reference radius of a halo.

    .. note:: This is ``R = (3M/(4*pi*rho_c(a)*Delta))^(1/3)``,
              where rho_c is the critical matter density at scale factor ``a``.

    Arguments
    ---------
    cosmo : ``pyccl.Cosmology`` object
        Cosmological parameters.
    halo_mass : float or array_like
        Halo mass [Msun].
    a : float
        Scale factor
    Delta : float
        Overdensity parameter.
    is_matter : boolean
        True when R_Delta is calculated using the average matter density.
        False when R_Delta is calculated using the critical density.

    Returns
    -------
    float or array_like : The halo reference radius in `Mpc`.
    """
    if is_matter: omega_factor = ccl.omega_x(cosmo, a, "matter")
    else: omega_factor = 1

    c1 = (cosmo["h"] * ccl.h_over_h0(cosmo, a))**2
    prefac = 1.16217766e12 * Delta * omega_factor * c1

    return (halo_mass/prefac)**(1/3)



def max_multipole(fname, cosmo, Rmax=1):
    z, N = np.loadtxt(fname, unpack=True)
    N *= len(N)/simps(N, x=z)  # normalise histogram
    z_avg = np.average(z, weights=N)
    chi_avg = ccl.comoving_radial_distance(cosmo, 1/(1+z_avg))
    lmax = 1/Rmax * chi_avg - 1/2
    return lmax



def dataman(cosmo, datasets, covars, dndz):
    """
    Constructs data vectors from different data sets and computes their inverse
    covariance matrix.

    .. note:: At the moment, it works with equally sized data sets.

    Arguments
    ---------
    datasets : list of strings
        The names of the surveys being cross-correlated, e.g. ``["2mpz_2mpz"]``.
    covars : list of strings
        The names of the covariances being cross-correlated.
        Must be ``len(datasets)**2``.
    dndz : str
        The name of the number density histogram for the survey used.

    """
    # check input
    if len(covars) != len(datasets)**2:
        raise NameError("Number of input covariances should be the square \
                        of the number of input datasets")
    # data directories #
    dir1 = "../analysis/data/dndz/"         # dndz
    dir2 = "../analysis/out_ns512_linlog/"  # Cl, dCl

    # dndz #
    dndz = dir1 + dndz + ".txt"
    # science #
    data = [np.load(dir2 + "cl_" + d + ".npz") for d in datasets]
    # covariances #
    cov = [np.load(dir2 + "cov_" + c + ".npz") for c in covars]


    l_arr, cl_arr = [np.zeros(0) for i in range(2)]
    for d in data:
        # x-data
        l = d["leff"]
        mask = l < max_multipole(dndz, cosmo)
        l_arr = np.append(l_arr, l[mask])
        # y-data
        cl_arr = np.append(cl_arr, (d["cell"] - d["nell"])[mask])


    # covariances
    # Individual covars are passed first across columns and then across rows.
    # Extract covariance from dataset and mask it, then stack them across
    # their columns in groups. Then, stack the groups across their rows
    # to complete the square matrix.
    covar = []
    for i in len(datasets)*np.arange(len(datasets)):
        covar.append(np.column_stack((
                [  (c["cov"])[mask, :][:, mask]
                for c in cov[i : i+len(datasets)] ] )) )
    covar = np.vstack(([i for i in covar]))
    I = np.linalg.inv(covar)

    return l_arr, cl_arr, I