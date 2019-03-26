import pymaster as nmt
import numpy as np
import healpy as hp


class Field(object):
    """
    Field creator.

    Args:
        nside (int): HEALPix resolution parameter.
        name (str): field name.
        mask_id (str): ID for this mask.
        fname_mask (str): path to file containing the mask.
        fname_map (str): path to file containing the sky map.
        fname_dndz (str): path to file containing the redshift
            distribution for this field. Pass `None` if not
            relevant.
        field_mask (int): HDU in which the mask is stored.
        field_map (int): HDU in which the map is stored.
        is_ndens (bool): set to True if this is a number
            density tracer.
    """
    def __init__(self, nside, name, mask_id,
                 fname_mask, fname_map, fname_dndz,
                 field_mask=0, field_map=0, is_ndens=True):
        self.name = name
        self.nside = nside
        self.mask_id = mask_id
        self.fname_mask = fname_mask
        self.is_ndens = is_ndens
        self.mask = hp.ud_grade(hp.read_map(fname_mask, verbose=False,
                                            field=field_mask), nside_out=nside)
        self.map0 = hp.ud_grade(hp.read_map(fname_map, verbose=False,
                                            field=field_map), nside_out=nside)
        mask_bn = np.ones_like(self.mask)
        mask_bn[self.mask <= 0] = 0  # Binary mask
        self.map0 *= mask_bn  # Remove masked pixels
        if is_ndens:  # Compute delta if this is a number density map
            mean_g = np.sum(self.map0*self.mask) / np.sum(self.mask)
            self.ndens = mean_g * hp.nside2npix(self.nside) / (4*np.pi)
            self.map = self.mask*(self.map0 / mean_g - 1.)
            self.dndz = fname_dndz
            self.z, self.nz = np.loadtxt(self.dndz, unpack=True)
            z_inrange = self.z[(self.nz > 0.005 * np.amax(self.nz))]
            self.zrange = np.array([z_inrange[0], z_inrange[-1]])
        else:
            self.ndens = 0
            self.map = self.map0
            self.z = None
            self.dndz = None

        self.field = nmt.NmtField(self.mask, [self.map])

    def update_field(self, new_mask=1.):
        """
        Updates the `NmtField` object stored in this `Field`
        multiplying the original mask by a new one. Note that
        this does not overwrite the original mask or
        map stored here, only the `NmtField` object.

        Args:
            new_mask (float or array): new mask.
        """
        self.field = nmt.NmtField(self.mask * new_mask, [self.map])
