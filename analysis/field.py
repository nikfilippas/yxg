import pymaster as nmt
import numpy as np
import healpy as hp

class Field(object):
    def __init__(self,nside,name,mask_id,
                 fname_mask,fname_map,fname_dndz,
                 field_mask=0,field_map=0,is_ndens=True):
        self.name=name
        self.nside=nside
        self.mask_id=mask_id
        self.fname_mask=fname_mask
        self.is_ndens=is_ndens
        self.mask=hp.ud_grade(hp.read_map(fname_mask,verbose=False,field=field_mask),nside_out=nside)
        self.map0=hp.ud_grade(hp.read_map(fname_map,verbose=False,field=field_map),nside_out=nside)
        mask_bn=np.ones_like(self.mask);
        mask_bn[self.mask<=0]=0; #Binary mask
        self.map0*=mask_bn #Remove masked pixels
        if is_ndens: #Compute delta if this is a number density map
            mean_g=np.sum(self.map0*self.mask)/np.sum(self.mask)
            self.ndens=mean_g*hp.nside2npix(self.nside)/(4*np.pi)
            self.map=self.mask*(self.map0/mean_g-1.)
            self.dndz=fname_dndz
            self.z,self.nz=np.loadtxt(self.dndz,unpack=True)
            z_inrange=self.z[(self.nz>0.005*np.amax(self.nz))]
            self.zrange=np.array([z_inrange[0],z_inrange[-1]])
        else:
            self.ndens=0
            self.map=self.map0
            self.z=None
            self.dndz=None

        self.field=nmt.NmtField(self.mask,[self.map])
        
    def update_field(self,new_mask=1.):
        self.field=nmt.NmtField(self.mask*new_mask,[self.map]) 
