"""
This script calculates the 1h- and 2h- halo contribution of any two profiles.
"""


import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import yxg



dndm = 1

q1 = yxg.Profile("arnaud")
q2 = yxg.Profile("arnaud")

k = np.logspace(-1, 1, 100)

M = 1e14
z = 0

U = q1.fourier_profile(k, M, z)
V = q2.fourier_profile(k, M, z)



def integrand(x, q):
    I = form_factor(x)*x**2*np.sinc(q*x)
    return I

q_array = np.logspace(-3, 3, 1000)
f_array = [quad(integrand, 0, np.inf, args=q)[0] for q in q_array]

F = interp1d(q_array, np.array(f_array), fill_value=0)
