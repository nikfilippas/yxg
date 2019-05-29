from healpy.fitsfunc import read_map
from healpy.visufunc import mollview
from matplotlib import cm
import matplotlib.pyplot as plt

dir1 = "../../data/maps/"

f1 = dir1 + "mask_planck60.fits"
f2 = dir1 + "mask_planck60S.fits"

data1 = read_map(f1)
data2 = read_map(f2)

mollview(data1+data2, title="", cbar=False, cmap=cm.gist_heat_r)

plt.savefig("mask.pdf", bbox_inches="tight")