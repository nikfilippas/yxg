"""
"""

import numpy as np
import matplotlib.pyplot as plt

D = np.loadtxt("data/cell_nick.txt", skiprows=1)


l, Cl = D[:,0], D[:,1]

Cl *= 1e12*l*(l+1)/(2*np.pi)


plt.loglog(l, Cl)
