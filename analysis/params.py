import yaml
import pyccl as ccl
from .bandpowers import Bandpowers

class ParamRun(object):
    def __init__(self,fname):
        with open(fname) as f:
            self.p=yaml.safe_load(f)

    def get_cosmo(self):
        return ccl.Cosmology(Omega_c=0.26066676,
                             Omega_b=0.048974682,
                             h=0.6766,
                             sigma8=0.8102,
                             n_s=0.9665)

    def get_outdir(self):
        return self.p['global']['output_dir']

    def get_bandpowers(self):
        return Bandpowers(self.p['global']['nside'],
                          self.p['bandpowers'])

    def get_models(self):
        models={}
        for d in self.p['maps']:
            models[d['name']]=d.get('model')
        return models

    def get_fname_mcm(self,f1,f2,jk_region=None):
        fname=self.get_outdir()+"/mcm_"+f1.mask_id+"_"+f2.mask_id
        if jk_region is not None:
            fname+="_jk%d"%jk_region
        fname+=".mcm"
        return fname

    def get_prefix_cls(self,f1,f2):
        return self.get_outdir()+"/cls_"+f1.name+"_"+f2.name
    
    def get_fname_cls(self,f1,f2,jk_region=None):
        fname=self.get_prefix_cls(f1,f2)
        if jk_region is not None:
            fname+="_jk%d"%jk_region
        fname+=".npz"
        return fname

    def get_fname_cmcm(self,f1,f2,f3,f4):
        fname=self.get_outdir()+"/cmcm_"
        fname+=f1.mask_id+"_"
        fname+=f2.mask_id+"_"
        fname+=f3.mask_id+"_"
        fname+=f4.mask_id+".cmcm"
        return fname

    def get_fname_cov(self,f1,f2,f3,f4,suffix):
        fname=self.get_outdir()+"/cov_"+suffix+"_"
        fname+=f1.name+"_"
        fname+=f2.name+"_"
        fname+=f3.name+"_"
        fname+=f4.name+".npz"
        return fname

    def get(self,k):
        return self.p.get(k)

    def do_jk(self):
        return self.p['jk']['do']
    
    def get_nside(self):
        return self.p['global']['nside']
