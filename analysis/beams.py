import numpy as np

def beam_y_planck(l):
    sigma_y_rad=np.radians(10./2.355/60)
    return np.exp(-0.5*l*(l+1)*sigma_y_rad**2)

def beam_hpix(l,ns):
    fwhm_hp_amin=60*41.7/ns
    sigma_hp_rad=np.radians(fwhm_hp_amin/2.355/60)
    #return np.exp(-0.5*l*(l+1)*sigma_hp_rad**2)
    return np.ones_like(l)
