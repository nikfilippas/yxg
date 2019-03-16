import sys
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import pyccl as ccl

import fittingtools as ft



# SURVEY PROPERTIES #
dir1 = "../analysis/data/dndz/"
wisc = ["wisc_b%d" % i for i in range(1, 6)]
surveys = ["2mpz"] + wisc
bins = np.append([1], np.arange(1, 6))
sprops = ft.survey_properties(dir1, surveys, bins)

cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665)



# PARAMETER : [VALUE, STATUS, CONSTRAINTS] #
# (free : 0) ::: (fixed : 1) ::: (coupled : -N)
priors = {"Mmin"       :  [12.0,   -1,   (10, 16)],
          "M0"         :  [12.0,   -1,   (10, 16)],
          "M1"         :  [13.5,    0,   (10, 16)],
          "sigma_lnM"  :  [0.15,    1,   (0.1, 1.0)],
          "alpha"      :  [1.0,     1,   (0.5, 1.5)],
          "fc"         :  [1.0,     1,   (0.1, 1.0)],
          "bg"         :  [1.0,     1,   (0, np.inf)],
          "bmax"       :  [1.0,     1,   (0, np.inf)],
          "r_corr"     :  [0.0,     0,   (-1, 1)],
          "b_hydro"    :  [0.50,    0,   (0.1, 0.9)]}



# INPUT HANDLING #
if len(sys.argv) == 1:
    y = input("Warning: Chains will start from the beginning!\nContinue? (y/n): ")
    if y == "y":
        c = False
        # minimizer
        minimizer = lambda sur: ft.param_fiducial(sur, sprops, cosmo, priors, v=True)
        p0 = Pool().map(minimizer, list(sprops.keys()))
        sys.argv.append(None)
    elif y == "n":
        y = input("Start chains from saved states? (y/n): ")
        if y == "y":
            sys.argv.append("continue")

if sys.argv[1] == "continue":
    c = True
    p0 = [priors for sur in sprops]
    print("Continuing from saved samplers...")
else:
    raise ValueError("To continue from saved backends, use 'continue' keyword.")


# MCMC
sampler = lambda sur, p: ft.MCMC(sur, sprops, cosmo, p,
                                 nwalkers=200, nsteps=1000, continued=c, v=False)
results = Pool().map(sampler, list(sprops.keys()), p0)