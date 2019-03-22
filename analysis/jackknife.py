import numpy as np
import healpy as hp

class JackKnife(object):
    def __init__(self,nside_jk,mask,frac_thr=0.5):
        npatch=hp.nside2npix(nside_jk)
        self.nside_maps=hp.npix2nside(len(mask))
        jk_ids=hp.ud_grade(np.arange(npatch),nside_out=self.nside_maps).astype(int)
        self.npix_per_patch=(self.nside_maps//nside_jk)**2
        ipix=np.arange(hp.nside2npix(self.nside_maps))

        jk_pixels=[]
        for ip in range(npatch):
            msk=jk_ids==ip
            frac=np.sum(mask[msk])/self.npix_per_patch
            if frac>frac_thr:
                jk_pixels.append(ipix[msk])
        self.jk_pixels=np.array(jk_pixels)
        self.npatches=len(self.jk_pixels)

    def get_jk_mask(self,jk_id):
        if jk_id>=self.npatches:
            raise ValueError("Asking for non-existent jackknife region")
        msk=np.ones(hp.nside2npix(self.nside_maps))
        msk[self.jk_pixels[jk_id]]=0
        return msk
