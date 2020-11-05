#### yxg reproduction code ####
import numpy as np
from analysis.params import ParamRun
from model.profile2D import HOD, Arnaud
from model.trispectrum import hm_1h_trispectrum

fname = "params_lensing.yml"
p = ParamRun(fname)
p.p["mcmc"]["mfunc"] = "tinker"
cosmo = p.get_cosmo()

for m in p.get("maps"):
    if m["name"] == "wisc3":
        mg = m
    elif m["name"] == "y_milca":
        my = m
        break

g = HOD(nz_file=mg["dndz"])
y = Arnaud()

k = np.logspace(-3, 2, 256)
a = 1/(1+g.z_avg)

kwargs = mg["model"]
kwargs["Mmin"] = kwargs["lMmin"]
kwargs["M0"] = kwargs["lM0"]
kwargs["M1"] = kwargs["lM1"]
kwargs["sigma_lnM"] = kwargs["sigmaLogM"]
kwargs["r_corr"] = kwargs["r_corr_gy"]

tri_gggg = hm_1h_trispectrum(cosmo, k, a, (g,g,g,g), **kwargs)
tri_gggy = hm_1h_trispectrum(cosmo, k, a, (g,g,g,y), **kwargs)
tri_gygy = hm_1h_trispectrum(cosmo, k, a, (g,y,g,y), **kwargs)
tri_yyyy = hm_1h_trispectrum(cosmo, k, a, (y,y,y,y), **kwargs)

np.savez("../yxgxk/tri",
         tri_gggg=tri_gggg,
         tri_gggy=tri_gggy,
         tri_gygy=tri_gygy,
         tri_yyyy=tri_yyyy)


###########################
from model.trispectrum import hm_ang_1h_covariance


fsky = 0.6
l = np.arange(6, 2500, 10)
zrange_g = np.array([5e-04, 5.985e-01])
zrange_y = np.array([1e-6, 6])

cov_gggg = hm_ang_1h_covariance(cosmo, fsky, l, (g,g), (g,g),
                                zrange_a=zrange_g, zpoints_a=64,
                                zrange_b=zrange_g, zpoints_b=64,
                                **kwargs)