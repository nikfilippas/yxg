"""test code"""
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from matplotlib.cm import viridis

import profile2D
import pspec



def dataplot(cosmo, prof1, prof2, xdata, ydata, yerr, popt, color):
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
    kwargs = dict(zip(params, popt))
    Cl = pspec.ang_power_spectrum(cosmo, xdata, (prof1, prof2),
                                  zrange=(0.001, 0.3), zpoints=64,
                                  is_zlog=False, **kwargs)
    Cl *= B(xdata, sigma)

    plt.loglog(xdata, Cl, c=color)

B = lambda l, sigma: np.exp(-(l*(l+1)/2)*sigma**2)
sigma = np.deg2rad(10/60) / (2*np.sqrt(2*np.log(2)))


## DATA ##
# science
#data = np.load("../analysis/out_ns512_linlog/cl_2mpz_2mpz.npz")     # clgg
data = np.load("../../analysis/out_ns512_linlog/cl_2mpz_y_milca.npz")  # clyg
# covariances
#cov = np.load("../analysis/out_ns512_linlog/cov_2mpz_2mpz_2mpz_2mpz.npz")        # clgg
cov = np.load("../../analysis/out_ns512_linlog/cov_2mpz_y_milca_2mpz_y_milca.npz")  # clyg

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


## PROFILE ##
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)

nz = "../../analysis/data/dndz/2MPZ_bin1.txt"
prof1 = profile2D.HOD(nz_file=nz)
prof2 = profile2D.Arnaud()


p0 = np.array([12.3, 12.5, 13.65, 0.5, 1.1, 0.8])

b_test = np.arange(0.05, 0.95, 0.05)
p_trial = [np.append(p0, b) for i, b in enumerate(b_test)]


plt.figure()
plt.loglog()
plt.title("$b_{hydro}$ from $0.05$ to $0.95$ in increments of $0.05$")
plt.xlabel("$\\ell$",fontsize=15)
plt.ylabel("$C_\\ell$",fontsize=15)
plt.errorbar(l, cl, err, fmt="r.", ms=5)

col = [viridis(i) for i in np.linspace(0, 0.9, len(b_test))]
_ = [dataplot(cosmo, prof1, prof2, l, cl, err, p_trial[i], col[i]) for i, _ in enumerate(b_test)]
plt.show()
