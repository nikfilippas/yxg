import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

import pspec
import profile2D as p2D

import pspec2
import profile2D2 as p2D2

cosmo = ccl.Cosmology(Omega_c=0.26066676, Omega_b=0.048974682, h=0.6766, sigma8=0.8102, n_s=0.9665)
l = np.geomspace(6, 1300, 1000)


sprops = {"2mpz"     :  [(0.0005, 0.0745), 1],
          "wisc_b1"  :  [(0.0005, 0.1405), 1],
          "wisc_b2"  :  [(0.0215, 0.1575), 2],
          "wisc_b3"  :  [(0.0505, 0.1785), 3],
          "wisc_b4"  :  [(0.0735, 0.2035), 4],
          "wisc_b5"  :  [(0.0895, 0.2355), 5]}

dir1 = "../analysis/data/dndz/"

params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
popt = [11.99, 14.94, 13.18, 0.26, 1.43, 0.54, 0.45]
kwargs = dict(zip(params, popt))

Cl, C1h, C2h = [[np.zeros_like(l) for j in list(sprops.keys())] for i in range(3)]
cl, c1h, c2h = [[np.zeros_like(l) for j in list(sprops.keys())] for i in range(3)]
for i, sur in enumerate(list(sprops.keys())):
    key = sur.strip("_b%d" % sprops[sur][1])
    fname = dir1 + key.upper() + "_bin%d" % sprops[sur][1] + ".txt"

    p1 = p2D.Arnaud()
#    p1 = p2D.HOD(nz_file=fname)
    p2 = p2D.HOD(nz_file=fname)
    profiles = (p1, p2)

    Cl[i] = pspec.ang_power_spectrum(cosmo, l, profiles, **kwargs)
    C1h[i] = pspec.ang_power_spectrum(cosmo, l, profiles, include_1h=True, include_2h=False, **kwargs)
    C2h[i] = pspec.ang_power_spectrum(cosmo, l, profiles, include_1h=False, include_2h=True, **kwargs)



    p1 = p2D2.Arnaud()
#    p1 = p2D2.HOD(nz_file=fname)
    p2 = p2D2.HOD(nz_file=fname)
    profiles = (p1, p2)

    cl[i] = pspec2.ang_power_spectrum(cosmo, l, profiles, **kwargs)
    c1h[i] = pspec2.ang_power_spectrum(cosmo, l, profiles, include_1h=True, include_2h=False, **kwargs)
    c2h[i] = pspec2.ang_power_spectrum(cosmo, l, profiles, include_1h=False, include_2h=True, **kwargs)



fig, ax = plt.subplots(2, 6, sharex=True, sharey=True, figsize=(20,10))

for i, axis in enumerate(ax[0]):
    axis.loglog(l, Cl[i], "b")
    axis.loglog(l, C1h[i], "r")
    axis.loglog(l, C2h[i], "g")


for i, axis in enumerate(ax[1]):
    axis.loglog(l, cl[i], "b")
    axis.loglog(l, c1h[i], "r")
    axis.loglog(l, c2h[i], "g")

plt.title("clyg")
plt.tight_layout()

fig.savefig("old_new_clyg.pdf", bbox_inchec="tight")