import numpy as np


def beam_gaussian(l, fwhm_amin):
    """
    Returns the SHT of a Gaussian beam.

    Args:
        l (float or array): multipoles.
        fwhm_amin (float): full-widht half-max in arcmins.

    Returns:
        float or array: beam sampled at `l`.
    """
    sigma_rad = np.radians(fwhm_amin / 2.355 / 60)
    return np.exp(-0.5 * l * (l + 1) * sigma_rad**2)


def beam_hpix(l, ns):
    """
    Returns the SHT of the beam associated with a HEALPix
    pixel size.

    Args:
        l (float or array): multipoles.
        ns (int): HEALPix resolution parameter.

    Returns:
        float or array: beam sampled at `l`.
    """
    fwhm_hp_amin = 60 * 41.7 / ns
    return beam_gaussian(l, fwhm_hp_amin)
