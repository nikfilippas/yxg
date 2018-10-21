"""
This script examines the ratio of the Arnaud profile if we include the radial
dependence of mass in the power law (alpha-P-prime of x), and if we exclude it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis as cmap
from profiles import Arnaud



def Arnaud_ratio(x, M, z):
    """Computes the ratio of two Arnaud profiles of a given mass and at a given
    redshift. The profiles are identical, except one uses the full Arnaud profile
    including a radial mass dependence.
    """
    r = Arnaud(x, M, z, aPP=False) / Arnaud(x, M, z, aPP=True)
    return r


savedir = "images/"  # output directory

x = np.logspace(-3, np.log10(2), 1000)  # R/R_Delta

M_array = np.logspace(np.log10(6.0e13), np.log10(1.3e15), 8)
M_dep = [Arnaud_ratio(x, M=M, z=0) for M in M_array]


fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.set_xlabel("$R \mathrm{ /R_{500} }$", fontsize=16)
lbl = r"$\frac{ P_{th}\left( x, M_{500}, z \right) }{ P_{th}\left( x, M_{500}, z, \alpha(x) \right) }$"
ax.set_ylabel(lbl, fontsize=16)

ax.axhline(y=1, ls="--", c="k")

colors = cmap(np.linspace(0, 1, len(M_array)))
for i, _ in enumerate(M_array):
    txt = r"$ %.1e\ \mathrm{M_{\odot}}$" % M_array[i]
    txt = txt[:5] + r"\times 10^{%s}" % txt[7:9] + txt[10:]
    ax.plot(x, M_dep[i], lw=2, c=colors[i], label=txt)

ax.margins(x=0)
ax.legend(loc="lower right", ncol=2, fontsize=10)
fig.savefig(savedir+"arnaud_ratio.pdf", bbox_inches="tight")
plt.close()
