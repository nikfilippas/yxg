import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import zipfile
import itertools
import os


M = np.logspace(10, 15, 100)
cosmo = ccl.Cosmology(Omega_c=0.2678,
                      Omega_b=0.049,
                      h=0.6704,
                      sigma8=0.8347,
                      n_s=0.9619)
h = cosmo["h"]

datadir = "mfuncs/"
num1 = ["08", "10"]
num2 = ["001", "005", "010", "050", "100", "200"]

for mfname in itertools.product(*(num1, num2)):
    name = "T" + mfname[0] + "z" + mfname[1]
    archive = zipfile.ZipFile(datadir + name + ".zip", "r")
    _ = archive.extract("mVector_" + name + " .txt", path=datadir)


for f in os.listdir(datadir):
    if f[-3:] == "txt":
        data = np.loadtxt(datadir + f, skiprows=12, usecols=(0, 7))
        x, y = data.T

        z = int(f[-8: -5]) / 100
        massfunc = ccl.massfunc(cosmo, M, 1/(1+z), overdensity=500)

        plt.loglog(x/h, y*h**3, "b-", lw=3)
        plt.loglog(M, massfunc, "r-", lw=3)
        plt.xlabel(r"$M_{\odot}$", fontsize=16)
        plt.ylabel(r"$\frac{dn}{d \ \log_{10}(M_{\odot})}$", fontsize=16)
        plt.savefig(datadir + "plots/" + f[8:-5] + ".pdf", bbox_inches="tight")
        plt.close()
