"""
This script contains definitions of useful fitting functions for quick
retrieval and data analysis.
"""

from collections import ChainMap
from itertools import product
import numpy as np
import pyccl as ccl
import emcee
from scipy.optimize import minimize

import profile2D
import pspec



def max_multipole(fname, cosmo, kmax=1):
    """Calculates the highest working multipole for HOD cross-correlation."""
    z, N = np.loadtxt(fname, unpack=True)
    z_avg = np.average(z, weights=N)
    chi_avg = ccl.comoving_radial_distance(cosmo, 1/(1+z_avg))
    lmax = kmax*chi_avg - 1/2
    return lmax



def survey_properties(directory, surveys, bins, cutoff=0.5):
    """Returns a ``dict`` of the survey ``zrange`` and its bin."""
    Z = [[] for i, _ in enumerate(surveys)]
    for i, sur in enumerate(surveys):
        fname = directory + sur.strip("_b%d" % i).upper() + "_bin%d.txt" % bins[i]
        z, N = np.loadtxt(fname, unpack=True)
        m = N.argmax()
        imin = np.abs(N[:m] - cutoff*N.max()/100).argmin()
        imax = np.abs(N[m:] - cutoff*N.max()/100).argmin()
        Z[i] = (z[imin], z[m+imax])

    sprops = dict(zip(surveys, zip(Z, bins)))
    return sprops



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
    def Beam(l, profile, arcmin=10):
        """Calculates the beam of a data set."""
        if profile == "y":
            sigma = np.deg2rad(arcmin/2.355/60)
            B = np.exp(-0.5*l*(l+1)*sigma**2)
        else:
            B = np.ones_like(l)
        return B

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

    # Unpack science data & determine profile+beam
    l_arr, cl_arr, dcl_arr, mask = [[[] for C in cells] for i in range(4)]
    profiles, beams = [[[] for C in cells] for i in range(2)]
    for i, (d, C) in enumerate(zip(data, cells)):
        keys = [x.strip() for x in C.split(",")]

        # retrieve dN/dz if HOD profile is computed
        if any([param[key] == "g" for key in keys]):
            if z_bin is None:
                raise TypeError("You must define a redshift bin.")

            idx = np.where([param[key] == "g" for key in keys])[0][0]
            key = keys[idx].strip("_b%d" % z_bin)
            dndz = dir1 + key.upper() + "_bin%d" % z_bin + ".txt"

        # x-data
        l = d["leff"]
        mask[i] = l < max_multipole(dndz, cosmo, kmax=1)
        l_arr[i] = l[mask[i]]
        # y-data
        dcl_arr[i] = d["nell"][mask[i]]
        cl_arr[i] = d["cell"][mask[i]] - dcl_arr[i]

        for key in keys:
            if param[key] == "y":  # tSZ
                profiles[i].append(profile2D.Arnaud())
                beams[i].append(Beam(l_arr[i], "y"))
            if param[key] == "g":  # galaxy density
                profiles[i].append(profile2D.HOD(nz_file=dndz))
                beams[i].append(Beam(l_arr[i], "g"))

    # Unpacking covariances
    grid = list(product(range(len(data)), range(len(data))))  # coordinate grid
    covar = [[] for c in cov]
    for i, c in enumerate(cov):
        # pick mask corresponding to coordinate grid
        covar[i] = (c["cov"])[mask[grid[i][0]], :][:, mask[grid[i][1]]]

    # Concatenate
    c = [[] for i in data]
    for i, _ in enumerate(data):
        idx = list(map(int, len(data)*i + np.arange(len(data))))  # group indices
        c[i] = np.column_stack(([covar[j] for j in idx]))         # stack vertically
    covar = np.vstack((c))                                        # stack horizontally
    I = np.linalg.inv(covar)

    return l_arr, cl_arr, I, profiles, beams



def split_kwargs(**priors):
    """Splits priors into free, fixed, and coupled parameters."""
    cvals = list(set([priors[key][1] for key in sorted(priors) if priors[key][1] < 0]))
    coupled = [{key: priors[key] for key in sorted(priors) if priors[key][1] == val} for val in cvals]

    for c in coupled:
        vals = np.array(list(val[0] for val in list(c.values())))
        try:
            assert (vals == vals[0]).all()
        except AssertionError:
            raise(AssertionError("Declared coupled parameters don't match!"))

    free = {key: priors[key] for key in sorted(priors) if priors[key][1] == 0}
    fixed = {key: priors[key] for key in sorted(priors) if priors[key][1] == 1}
    return free, fixed, coupled



def build_kwargs(popt, free, fixed, coupled, full=True):
    """Reconstructs the posterior parameters."""
    # update free parameters
    for i, key in enumerate(sorted(free)):
        free[key][0] = popt[i]
    # update coupled parameters
    for i, c in enumerate(coupled):
        for key in c:
            c[key][0] = popt[len(free)+i]

    # re-build kwargs
    kwargs = dict(ChainMap(*coupled), **free, **fixed)
    if not full: kwargs = {key: kwargs[key][0] for key in kwargs}
    return kwargs



def lnprior(**priors):
    """Priors."""
    fitpar = {key: priors[key] for key in priors if priors[key][1] != 1}
    test = all([val[2][0] <= val[0] <= val[2][1] for val in fitpar.values()])
    return 0.0 if test else -np.inf



def lnprob(theta, setup, lnprior=None, negative=False, v=True):
    """Posterior probability distribution to be sampled."""
    # extract parameters
    cosmo = setup["cosmo"]
    prof = setup["profiles"]
    beams = setup["beams"]
    l_arr = setup["l_arr"]
    cl_arr = setup["cl_arr"]
    I = setup["inv_covar"]
    zrange = setup["zrange"]

    free = setup["free"]
    fixed = setup["fixed"]
    coupled = setup["coupled"]
    kwargs = build_kwargs(theta, free, fixed, coupled, full=False)

    prior_test = dict(ChainMap(*coupled), **free)
    lp = lnprior(**prior_test) if lnprior is not None else 0.0
    # Piecewise probability handling
    if not np.isfinite(lp):
        lnprob = -np.inf
    else:
        lnprob = lp  # only priors
        Cl = [pspec.ang_power_spectrum(cosmo, l, p, zrange, **kwargs
                            )*b[0]*b[1] for l, p, b in zip(l_arr, prof, beams)]
        cl, Cl = np.array(cl_arr).flatten(), np.array(Cl).flatten()  # data vectors

        # treat zero division (unphysical)
        if not Cl.all():
            lnprob = -np.inf
        else:
            lnprob += -0.5*np.dot(cl-Cl, np.dot(I, cl-Cl))

    if v:
        order = setup["order"]
        params = [kwargs[key] for key in order]
        global Neval
        print(Neval, params); Neval += 1

    return lnprob if not negative else -lnprob



def setup_run(survey, sprops, cosmo, priors):
    """Sets up the parameter finder run."""
    global Neval
    Neval = 1  # counter

    # Data Manipulation #
    data = [survey+","+survey, survey+","+"y_milca"]
    l, cl, I, prof, beams = dataman(data, z_bin=sprops[survey][1], cosmo=cosmo)
    setup = {"cosmo"     : cosmo,
             "profiles"  : prof,
             "beams"     : beams,
             "l_arr"     : l,
             "cl_arr"    : cl,
             "inv_covar" : I,
             "zrange"    : sprops[survey][0]}

    free, fixed, coupled = split_kwargs(**priors)

    # update setup dictionary
    setup["order"] = [key for key in priors if priors[key][1] != 1]
    setup["free"] = free
    setup["fixed"] = fixed
    setup["coupled"] = coupled

    p0 = np.array([val[0] for val in list(free.values())])
    p0 = np.append(p0, [list(par.values())[0][0] for par in coupled])

    return p0, setup



def MCMC(survey, sprops, cosmo, priors, nwalkers, nsteps, continued=False, v=False):
    """Runs the MCMC."""
    p0, setup = setup_run(survey, sprops, cosmo, priors)
    ndim = len(p0)

    # Set up backend for saving
    filename = "samplers/%s" % survey
    backend = emcee.backends.HDFBackend(filename)
    if not continued:
        backend.reset(nwalkers, ndim)
        pos = [p0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    else:
        pos = None

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(setup, lnprior, False, v),
                                    backend=backend)
    sampler.run_mcmc(pos, nsteps, store=True, progress=(not v))
    return sampler



def param_fiducial(survey, sprops, cosmo, priors, v):
    """Calculates a set of proposal parameters for the MCMC via a minimization."""
    p0, setup = setup_run(survey, sprops, cosmo, priors)
    free, fixed, coupled = setup["free"], setup["fixed"], setup["coupled"]

    res = minimize(lnprob, p0, args=(setup, lnprior, True, v), method="Powell")

    new_priors = build_kwargs(res.x, free, fixed, coupled)
    new_priors = {key: new_priors[key] for key in priors}

    return new_priors