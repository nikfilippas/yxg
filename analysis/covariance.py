import pymaster as nmt
import numpy as np
from .spectra import Spectrum
from scipy.interpolate import interp1d
    

class Covariance(object):
    def __init__(self,name1,name2,name3,name4,covariance):
        self.names=(name1,name2,name3,name4)
        self.covar=covariance

    def diag(self):
        return np.diag(self.covar)

    @classmethod
    def from_fields(Covariance,field_a1,field_a2,field_b1,field_b2,wsp_a,wsp_b,
                    cla1b1,cla1b2,cla2b1,cla2b2,cwsp=None):
        if cwsp is None:
            cwsp=nmt.NmtCovarianceWorskpace()
            cwsp.compute_coupling_coefficients(field_a1.field,field_a2.field,
                                               field_b1.field,field_b2.field)

        covar=nmt.gaussian_covariance(cwsp,0,0,0,0,[cla1b1],[cla1b2],[cla2b1],[cla2b2],wsp_a,wsp_b)
        return Covariance(field_a1.name,field_a2.name,field_b1.name,field_b2.name,covar)

    @classmethod
    def from_file(Covariance,fname,name1,name2,name3,name4):
        d=np.load(fname)
        return Covariance(name1,name2,name3,name4,d['cov'])

    def to_file(self,fname,n_samples=None):
        np.savez(fname[:-4], #Remove file suffix
                 cov=self.covar,n_samples=n_samples)

    @classmethod
    def from_jk(Covariance,njk,prefix1,prefix2,suffix,name1,name2,name3,name4):
        def get_fname(ii,jk_id):
            if ii==1:
                return prefix1+"%d"%jk_id+suffix
            else:
                return prefix2+"%d"%jk_id+suffix

        #Initialize data
        cls1=np.array([np.load(get_fname(1,jk_id))['cls'] for jk_id in range(njk)])
        cls2=np.array([np.load(get_fname(2,jk_id))['cls'] for jk_id in range(njk)])
        #Compute mean
        cls1_mean=np.mean(cls1,axis=0)
        cls2_mean=np.mean(cls2,axis=0)
        #Compute covariance
        cov=np.sum((cls1-cls1_mean[None,:])[:,:,None]*(cls2-cls2_mean[None,:])[:,None,:],axis=0)
        cov*=(njk-1.)/njk
        
        return Covariance(name1,name2,name3,name4,cov)

    @classmethod
    def from_options(Covariance,covars,cov_corr,cov_diag,covars2=None,cov_diag2=None):
        #Diag = MAX(diags)
        diag1=np.amax(np.array([cov.diag() for cov in covars]),axis=0)
        if covars2 is None:
            diag2=diag1
        else:
            diag2=np.amax(np.array([cov.diag() for cov in covars2]),axis=0)

        #Correlation matrix
        d1=np.diag(cov_diag.covar)
        if cov_diag2 is None:
            d2=d1
        else:
            d2=np.diag(cov_diag2.covar)
        corr=cov_corr.covar/np.sqrt(d1[:,None]*d2[None,:])

        #Joint covariance
        cov=corr*np.sqrt(diag1[:,None]*diag2[None,:])

        return Covariance(cov_corr.names[0],cov_corr.names[1],cov_corr.names[2],cov_corr.names[3],cov)
