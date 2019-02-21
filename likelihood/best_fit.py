import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl

import fittingtools as ft
import profile2D
import pspec



dir1 = "../analysis/data/dndz/"

# survey properties: name, zrange, bin
sprops = {"2mpz"   :  [(0.001, 0.300), 1],
          "wisc_b1":  [(0.001, 0.320), 1],
          "wisc_b2":  [(0.005, 0.370), 2],
          "wisc_b3":  [(0.020, 0.500), 3],
          "wisc_b4":  [(0.050, 0.600), 4],
          "wisc_b5":  [(0.070, 0.700), 5]}
surveys = list(sprops.keys())

cosmo = ccl.Cosmology(Omega_c=0.26066676, Omega_b=0.048974682, h=0.6766,
                      sigma8=0.8102, n_s=0.9665)


popt = np.array([np.load("fit_vals/"+sur+".npy") for sur in surveys])


# outer --> inner: (l,cl) --> (yg,gg) --> (surveys)
l, cl = [[[[] for s, _ in enumerate(surveys)] for i in range(2)] for i in range(2)]
for i, sur in enumerate(surveys):
    # data
    data = [sur+","+sur, sur+","+"y_milca"]
    l_arr, cl_arr, _, _ = ft.dataman(data, z_bin=sprops[sur][1], cosmo=cosmo)

    l[0][i], l[1][i] = l_arr
    cl[0][i], cl[1][i] = cl_arr



chi2, Np = [[[[] for s, _ in enumerate(surveys)] for i in range(2)] for i in range(2)]
Cl = [[[] for s, _ in enumerate(surveys)] for i in range(2)]   # 2 (yg, gg)
hal = [[[] for s, _ in enumerate(surveys)] for i in range(2)]  # 2 (1h, 2h)
p1 = profile2D.Arnaud()
for i, sur in enumerate(surveys):
    # model
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
    kwargs = dict(zip(params, popt[i][0]))
    # yxg
    Cl[0][i] = pspec.ang_power_spectrum(cosmo, l[0][i], (p1, p1), zrange=sprops[sur][0],
                      include_1h=True, include_2h=True, **kwargs)
    hal[0][i] = pspec.ang_power_spectrum(cosmo, l[0][i], (p1, p1), zrange=sprops[sur][0],
                       include_1h=True, include_2h=False, **kwargs)
    hal[1][i] = pspec.ang_power_spectrum(cosmo, l[0][i], (p1, p1), zrange=sprops[sur][0],
                       include_1h=False, include_2h=True, **kwargs)


    # gxg
    fname = dir1 + ("2MPZ_bin1.txt" if sur is "2mpz" else  sur[:4].upper() +
                    "_bin%d.txt" % sprops[sur][1])
    p2 = profile2D.HOD(nz_file=fname)
    Cl[1][i] = pspec.ang_power_spectrum(cosmo, l[1][i], (p1, p2), zrange=sprops[sur][0], **kwargs)


    # stats
    Np[0][i] = len(Cl[0][i])
    chi2[0][i] = np.sum((cl[0][i]-Cl[0][i])**2 / ()**2)  # FIXME: errors

    Np[1][i] = len(Cl[1][i])
    chi2[1][i] = np.sum((cl[1][i]-Cl[1][i])**2 / ()**2)  # FIXME: errors

    print(i+1, "/", len(surveys))


