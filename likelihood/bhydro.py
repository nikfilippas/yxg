import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt


# MCMC results: 50, 16, 84 quantiles
bh = np.array([[0.22159627, 0.28700594, 0.16940808, 0.14136074, 0.34945891, 0.59129961],
               [0.09937876, 0.17312259, 0.17563861, 0.26303459, 0.17669672, 0.11982896],
               [0.07129987, 0.13041275, 0.04213337, 0.03056129, 0.11163703, 0.05378206]])



dir1 = "../analysis/data/dndz/"

# g-surveys
#sdss = ["sdss_b%d" % i for i in range(1, 10)]
wisc = ["wisc_b%d" % i for i in range(1, 6)]
surveys = ["2mpz"] + wisc

z_arr = np.zeros((2, len(surveys)))
for i, s in enumerate(surveys):
    if s is "2mpz":
        fname = dir1 + "2MPZ_bin1.txt"
    else:
        fname = dir1 + s[:4].upper() + "_bin%d.txt" % int(s[6])
    z, N = np.loadtxt(fname, unpack=True)
    N *= len(N)/simps(N, x=z)  # normalise histogram
    z_arr[0, i] = np.average(z, weights=N)
    var = np.average((z-z_arr[0, i])**2, weights=N)
    z_arr[1, i] = np.sqrt(var)
