import numpy as np
import pyccl as ccl

import profile2D
import pspec

cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
l = np.arange(6, 1350, 1)
p1 = profile2D.Arnaud()
nzfile = '../../analysis/data/dndz/WISC_bin4.txt'
p2 = profile2D.HOD(nzfile)

params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
popt = [11.99, 14.94, 13.18, 0.26, 1.43, 0.54, 0.45]
kwargs = dict(zip(params, popt))


sur = "wisc_b5"

sprops = {"2mpz"   :  [(0.001, 0.300), (0.001, 0.300)],
          "wisc_b1":  [(0.001, 0.320), (0.001, 0.300)],
          "wisc_b2":  [(0.005, 0.370), (0.001, 0.300)],
          "wisc_b3":  [(0.020, 0.500), (0.001, 0.300)],
          "wisc_b4":  [(0.050, 0.600), (0.001, 0.300)],
          "wisc_b5":  [(0.001, 0.700), (0.001, 7.000)]}

X0 = pspec.ang_power_spectrum(cosmo, l, (p1, p2), zrange=sprops[sur][0],
                                 zpoints=32, is_zlog=True, **kwargs)

X1 = pspec.ang_power_spectrum(cosmo, l, (p1, p2), zrange=sprops[sur][0],
                                 zpoints=2048, is_zlog=True, **kwargs)

X2 = pspec.ang_power_spectrum(cosmo, l, (p1, p2), zrange=sprops[sur][1],
                                 zpoints=2048, is_zlog=False, **kwargs)

X3 = pspec.ang_power_spectrum(cosmo, l, (p1, p2), zrange=sprops[sur][1],
                                 zpoints=32, is_zlog=False, **kwargs)
