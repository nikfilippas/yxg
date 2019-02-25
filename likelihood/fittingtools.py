"""
This script contains definitions of useful fitting functions for quick
retrieval and data analysis.
"""

from itertools import product
import numpy as np
import pyccl as ccl
import emcee

import profile2D
import pspec



def max_multipole(fname, cosmo):
    """Calculates the highest working multipole for HOD cross-correlation."""
    z, N = np.loadtxt(fname, unpack=True)
    z_avg = np.average(z, weights=N)
    chi_avg = ccl.comoving_radial_distance(cosmo, 1/(1+z_avg))
    lmax = chi_avg - 1/2
    return lmax



def dataman(cells, z_bin=None, cosmo=None):
    """
    Constructs data vectors from different data sets and computes their inverse
    covariance matrix.

    Arguments
    ---------
    data : tuple or str
        The names of the surveys used to compute ``Cl``, in the form "s1, s2".
    bins : int
        The redshift bin used.
    cosmo : `pyccl.Cosmology` object
        Cosmological parameters.
    """
    # dictionary of surveys #
    sdss = ["sdss_b%d" % i for i in range(1, 10)]
    wisc = ["wisc_b%d" % i for i in range(1, 6)]
    param = {**dict.fromkeys(["2mpz"] + sdss + wisc,    "g"),
             **dict.fromkeys(["y_milca", "y_nilc"],     "y")}

    # data directories #
    dir1 = "../analysis/data/dndz/"         # dndz
    dir2 = "../analysis/out_ns512_linlog/"  # Cl, dCl

    # file names #
    cells = (cells,) if type(cells) == str else tuple(cells)  # convert to tuple
    dfiles  = ["_".join([x.strip() for x in C.split(",")]) for C in cells]
    cfiles = np.column_stack([x.flatten() for x in np.meshgrid(dfiles, dfiles)])
    cfiles = ["_".join(x) for x in cfiles]

    # load data #
    data = [np.load(dir2 + "cl_" + d + ".npz") for d in dfiles]   # science
    cov = [np.load(dir2 + "cov_" + c + ".npz") for c in cfiles]   # covariance

    # profiles #
    profiles = [[] for C in cells]
    for i, C in enumerate(cells):
        keys = [x.strip() for x in C.split(",")]
        for key in keys:
            if param[key] == "y":  # tSZ
                profiles[i].append(profile2D.Arnaud())
            if param[key] == "g":  # galaxy density
                if not z_bin:
                    raise TypeError("You must define a redshift bin.")
                if "dndz" not in locals():
                    key = key.strip("_b%d" % z_bin)
                    dndz = dir1 + key.upper() + "_bin%d" % z_bin + ".txt"
                profiles[i].append(profile2D.HOD(nz_file=dndz))

    # Unpacking science data
    l_arr, cl_arr, dcl_arr, mask = [[[] for C in cells] for i in range(4)]
    for i, d in enumerate(data):
        # x-data
        l = d["leff"]
        if z_bin: mask[i] = l < max_multipole(dndz, cosmo)
        l_arr[i] = l[mask[i]]
        # y-data
        dcl_arr[i] = d["nell"][mask[i]]
        cl_arr[i] = d["cell"][mask[i]] - dcl_arr[i]

    del dndz  # removing dndz from locals

    # Unpacking covariances
    grid = list(product(range(len(data)), range(len(data))))  # coordinate grid
    covar = [[] for c in cov]
    for i, c in enumerate(cov):
        # pick mask corresponding to coordinate grid
        covar[i] = (c["cov"])[mask[grid[i][0]], :][:, mask[grid[i][1]]]

    # Concatenate
    c = [[] for i in data]
    for i, _ in enumerate(data):
        idx = list(map(int, 2*i + np.arange(len(data))))   # group indices
        c[i] = np.column_stack(([covar[j] for j in idx]))  # stack vertically
    covar = np.vstack((c))                                 # stack horizontally
    I = np.linalg.inv(covar)

    return l_arr, cl_arr, I, profiles



def lnprior(theta):
    """Priors."""
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
    kwargs = dict(zip(params, theta))

    prior_test = (9 <= kwargs["Mmin"] <= 15)*\
                 (10 <= kwargs["M0"] <= 16)*\
                 (10 <= kwargs["M1"] <= 16)*\
                 (0.1 <= kwargs["sigma_lnM"] <= 1.0)*\
                 (0.5 <= kwargs["alpha"] <= 1.5)*\
                 (0.1 <= kwargs["fc"] <= 1.0)*\
                 (0.1 <= kwargs["b_hydro"] <= 0.9)*\
                 (kwargs["M0"] > kwargs["Mmin"])

    return 0.0 if prior_test else -np.inf



def lnprob(theta, lnprior=None, verbose=True, **setup):
    """Posterior probability distribution to be sampled."""
    if verbose:
        global Neval
        print("\r", Neval, theta); Neval += 1

    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
    kwargs = dict(zip(params, theta))

    cosmo = setup["cosmo"]
    prof = setup["profiles"]
    l_arr = setup["l_arr"]
    cl_arr = setup["cl_arr"]
    I = setup["inv_covar"]
    zrange = setup["zrange"]

    lp = lnprior(theta) if lnprior else 0.0
    # Piecewise probability handling
    if not np.isfinite(lp):
        lnprob = -np.inf
    else:
        lnprob = lp  # only priors
        Cl = [pspec.ang_power_spectrum(
                                cosmo, l, p, zrange, **kwargs
                                ) for l, p in zip(l_arr, prof)]
        cl, Cl = [np.array(x).flatten() for x in [cl_arr, Cl]]  # data vectors

        # treat zero division (unphysical)
        if not Cl.all():
            lnprob = -np.inf
        else:
            lnprob += -0.5*np.dot(cl-Cl, np.dot(I, cl-Cl))

    return lnprob



def MCMC(survey, sprops, cosmo, popt, lnprob, args, nwalkers=100, nsteps=500):
    """Runs the MCMC."""
    global Neval
    Neval = 1  # counter
    # Data Manipulation #
    data = [survey+","+survey, survey+","+"y_milca"]
    l, cl, I, prof = dataman(data, z_bin=sprops[survey][1], cosmo=cosmo)
    setup = {"cosmo"     : cosmo,
             "profiles"  : prof,
             "l_arr"     : l,
             "cl_arr"    : cl,
             "inv_covar" : I,
             "zrange"    : sprops[survey][0]}

    # MCMC #
    ndim = len(popt)
    pos = [popt + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=args, kwargs=setup)
    sampler.run_mcmc(pos, nsteps)
    return sampler