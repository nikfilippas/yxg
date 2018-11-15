"""
temp file for calcs and unstable test code
"""


import numpy as np
import pyccl as ccl

from cosmotools import R_Delta
import profile2D


# Cosmology Definition
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)

a = 1
Delta = 500
"""
rmin, rmax = 1e-4, 5  # [R_Delta]
kmin, kmax = 2*np.pi/rmax, 2*np.pi/rmin
# so, values of k probed, logarithmically span [kmin, kmax]
k_arr = np.logspace(np.log10(kmin), np.log10(kmax), 100)

p1 = profile2D.Arnaud()
p2 = profile2D.Arnaud()
p3 = profile2D.NFW()
"""

# Arnaud extrapolate
from scipy.integrate import quad
from scipy.interpolate import interp1d

def form_factor(x):
    c500 = 1.81
    alpha = 1.33
    beta = 4.13
    gama = 0.31

    f1 = (c500*x)**-gama
    f2 = (1+(c500*x)**alpha)**(-(beta-gama)/alpha)
    return f1*f2

def integrand(x):
    I = form_factor(x)*x
    return I

rrange=(1e-3, 10); qpoints=1e2

rmin, rmax = rrange  # physical distance [R_Delta]
qmin, qmax = 1/rmax, 1/rmin  # fourier space parameter

q1 = np.logspace(-5, np.log10(qmin))  # extrapolation low
q3 = np.logspace(np.log10(qmax), +5)  # extrapolation high

q_arr = np.logspace(np.log10(qmin), np.log10(qmax), qpoints)

f_arr = np.array([quad(integrand, a=1e-4, b=np.inf, weight="sin", wvar=q)[0] / q for q in q_arr])

F2 = interp1d(np.log10(q_arr), np.array(f_arr), kind="cubic", fill_value="extrapolate")
