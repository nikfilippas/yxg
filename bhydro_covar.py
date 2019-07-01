import numpy as np

save_dir = "output_default/"

p0 = np.load(save_dir + "b_hydro_jackknife.npy")
pj = [np.load(save_dir + "b_hydro_jackknife_jk%d.npy" % jk) for jk in np.arange(1, 461)]

pj = np.vstack(pj)