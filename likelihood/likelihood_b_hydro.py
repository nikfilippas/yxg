import numpy as np
from scipy.optimize import minimize
import pyccl as ccl

import pspec
import profile2D



def func(p0, args=None):
    """The function to be minimised."""
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
    if args is not None: p0 = np.hstack((args, p0))
    kwargs = dict(zip(params, p0))
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
                                      zrange=(0.001, 0.3), **kwargs)

        if Cl is None:  # treat zero division (unphysical)
            lnprob = -np.inf
        else:
            lnprob = -0.5*np.dot(cl-Cl, np.dot(I, cl-Cl))

    print(p0, -2*lnprob, len(l))  # output trial parameter values
    return -lnprob


Neval = 1  # display number of evaluations
def callbackf(X):
    global Neval
    print("{0:4d}".format(Neval), *X)
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


# Fixed HOD parameters
popt_fix = [12.00287818, 14.94087941, 13.18144554, 0.27649579, 1.43902899, 0.57055288]
p0 = 0.4
res_fix = minimize(func, p0, args=popt_fix, method="Powell", callback=callbackf)

# All parameters free
p0 = np.append(popt_fix, p0)
res_free = minimize(func, p0, method="Powell", callback=callbackf)



## PLOTS ##
import matplotlib.pyplot as plt

def dataplot(cosmo, prof1, prof2, xdata, ydata, yerr, popt_free, popt_fix, b_hydro):
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
    kwargs_free = dict(zip(params, popt_free))

    popt_fix = np.append(popt_fix, b_hydro)
    kwargs_fix = dict(zip(params, popt_fix))

    Cl_free, Cl_fix = [pspec.ang_power_spectrum(
                                cosmo, xdata, prof1, prof2,
                                zrange=(0.001, 0.3), **kwargs
                                ) for kwargs in [kwargs_free, kwargs_fix]]

    plt.figure()
    plt.xlabel("$\\ell$",fontsize=15)
    plt.ylabel("$C^{yg}_\\ell$",fontsize=15)

    plt.errorbar(xdata, ydata, yerr, fmt="rs")
    plt.loglog(xdata, Cl_fix, "k-", lw=3, label="HOD fixed, $b_{hydro} = %.2f$" % popt_fix[-1])
    plt.loglog(xdata, Cl_free, "k--", lw=3, label="HOD free, $b_{hydro} = %.2f$" % popt_free[-1])
    plt.legend(loc="upper right")
    plt.show()


dataplot(cosmo, prof1, prof2, l, cl, err, res_free.x, popt_fix, res_fix.x)
plt.savefig("../images/clyg_b_hydro_fit.pdf", dpi=600)


### RESULTS ##
## 7 free parameters
#kwargs = {"Mmin"      : 12.14154536,
#          "M0"        : 14.92770155,
#          "M1"        : 13.19313094,
#          "sigma_lnM" : 0.34787508,
#          "alpha"     : 1.45104188,
#          "fc"        : 0.56469074,
#          "b_hydro"   : 0.59489416}
#
## 1 free parameter (b_hydro)
#kwargs = {"Mmin"      : 12.00287818,
#          "M0"        : 14.94087941,
#          "M1"        : 13.18144554,
#          "sigma_lnM" : 0.27649579,
#          "alpha"     : 1.43902899,
#          "fc"        : 0.57055288,
#          "b_hydro"   : 0.41315248}
