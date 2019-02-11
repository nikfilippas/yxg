import numpy as np
from scipy.optimize import minimize
import pyccl as ccl

import pspec
import profile2D


def func(args):
    """The function to be minimised."""
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
    kwargs = dict(zip(params, args))
    global cosmo, prof1, prof2, l, cl

    prior_test = (6 <= kwargs["Mmin"] <= 17)*\
                 (6 <= kwargs["M0"] <= 17)*\
                 (6 <= kwargs["M1"] <= 17)*\
                 (0.1 <= kwargs["sigma_lnM"] <= 1.0)*\
                 (0.5 <= kwargs["alpha"] <= 1.5)*\
                 (0.1 <= kwargs["fc"] <= 1.0)*\
                 (0.1 <= kwargs["b_hydro"] <= 1.0)

    # Piecewise probability handling
    if not prior_test:
        lnprob = -np.inf
    else:
        Cl = pspec.ang_power_spectrum(cosmo, l, prof1, prof2,
                                      zrange=(0.001, 0.3), zpoints=64, **kwargs)

        if Cl is None:
            lnprob = -np.inf
        else:
            lnprob = -0.5*np.dot(cl-Cl, np.dot(I, cl-Cl))

    print(args, -2*lnprob, len(l))  # output trial parameter values
    return -lnprob


Neval = 1  # display number of evaluations
def callbackf(X):
    global Neval
    print("{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   \
          {4: 3.6f}   {5: 3.6f}   {6: 3.6f}".format(Neval, X[0], X[1], X[2],
                                                           X[3], X[4], X[5]))
    Neval += 1



## DATA ##
# science
data = np.load("../analysis/out_ns512_linlog/cl_2mpz_y_milca.npz")  # clyg
# covariances
cov = np.load("../analysis/out_ns512_linlog/cov_2mpz_y_milca_2mpz_y_milca.npz")  # clyg

# x-data
l = data["leff"]
mask = l < 260
l = l[mask]
# y-data
cl = data["cell"] - data["nell"]
cl = cl[mask]

# error bars
covar = cov["cov"]
covar = covar[mask, :][:, mask]
err = np.sqrt(np.diag(covar))



## MODEL 1 ##
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)

nz = "../analysis/data/dndz/2MPZ_bin1.txt"
prof1 = profile2D.HOD(nz_file=nz)
prof2 = profile2D.Arnaud()

I = np.linalg.inv(covar)  # inverse covariance


p0 = [12.00287818, 14.94087941, 13.18144554, 0.27649579, 1.43902899, 0.57055288, 0.4]
res = minimize(func, p0, method="Powell", callback=callbackf)



## PLOTS ##
import matplotlib.pyplot as plt

def dataplot(cosmo, prof1, prof2, xdata, ydata, yerr, popt):
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
    kwargs = dict(zip(params, popt))
    Cl = pspec.ang_power_spectrum(cosmo, xdata, prof1, prof2,
                                  zrange=(0.001, 0.3), zpoints=64, **kwargs)

    plt.figure()
    plt.xlabel("$\\ell$",fontsize=15)
    plt.ylabel("$C^{yg}_\\ell$",fontsize=15)

    plt.errorbar(xdata, ydata, yerr, fmt="rs")
    plt.loglog(xdata, Cl, "k-", lw=3, label="$b_{hydro} = 0.60$")
    plt.show()


dataplot(cosmo, prof1, prof2, l, cl, err, res.x)


#kwargs = {"Mmin"      : 12.1432452,
#          "M0"        : 14.92772773,
#          "M1"        : 13.19271811,
#          "sigma_lnM" : 0.35047501,
#          "alpha"     : 1.45163725,
#          "fc"        : 0.56485956,
#          "b_hydro"   : 0.59665216}
