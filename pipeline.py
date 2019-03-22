from __future__ import print_function
import numpy as np
import yaml
import sys
import os
import pymaster as nmt
import pyccl as ccl
import matplotlib.pyplot as plt
import healpy as hp
from scipy.interpolate import interp1d
from analysis.bandpowers import Bandpowers
from analysis.field import Field
from analysis.spectra import Spectrum
from analysis.covariance import Covariance
from analysis.beams import beam_y_planck, beam_hpix
from analysis.jackknife import JackKnife
from model.profile2D import Arnaud,HOD
from model.power_spectrum import hm_power_spectrum, hm_ang_power_spectrum
from model.trispectrum import hm_1h_trispectrum, hm_ang_1h_covariance

try:
    fname_params=sys.argv[1]
except:
    raise ValueError("Must provide param file name as command-line argument")

with open(fname_params) as f:
    p=yaml.safe_load(f)

#Read off N_side
nside=p['global']['nside']

#JackKnives setup
if p['jk']['do']:
    #Set union mask
    msk_tot=np.ones(hp.nside2npix(nside))
    for k in p['masks'].keys():
        msk_tot*=hp.ud_grade(hp.read_map(p['masks'][k],verbose=False),nside_out=nside)
    #Set jackknife regions
    jk=JackKnife(p['jk']['nside'],msk_tot)

#Cosmology (Planck 2018)
cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665)

#Create output directory if needed
os.system('mkdir -p '+p['global']['output_dir'])

#Generate bandpowers
print("Generating bandpowers")
bpw=Bandpowers(nside,p['bandpowers'])

#Generate all fields
print("Reading fields")
fields_ng=[]; fields_sz=[]; models={};
for d in p['maps']:
    print(" "+d['name'])
    f=Field(nside,d['name'],d['mask'],p['masks'][d['mask']],
            d['map'],d['dndz'],is_ndens=d['type']=='g')
    if f.is_ndens:
        fields_ng.append(f)
    else:
        fields_sz.append(f)
    models[d['name']]=d.get('model')

#Compute power spectra
print("Computing power spectra")
def get_fname_mcm(f1,f2,jk_region=None):
    fname=p['global']['output_dir']+"/mcm_"+f1.mask_id+"_"+f2.mask_id
    if jk_region is not None:
        fname+="_jk%d"%jk_region
    fname+=".mcm"
    return fname

def get_fname_cls(f1,f2,jk_region=None):
    fname=p['global']['output_dir']+"/cls_"+f1.name+"_"+f2.name
    if jk_region is not None:
        fname+="_jk%d"%jk_region
    fname+=".npz"
    return fname

def get_mcm(f1,f2,jk_region=None):
    fname=get_fname_mcm(f1,f2,jk_region=jk_region)
    mcm=nmt.NmtWorkspace()
    try:
        mcm.read_from(fname)
    except:
        print("  Computing MCM")
        mcm.compute_coupling_matrix(f1.field,f2.field,bpw.bn)
        mcm.write_to(fname)
    return mcm

def get_power_spectrum(f1,f2,jk_region=None,save_windows=True):
    print(" "+f1.name+","+f2.name)
    try:
        cls=Spectrum.from_file(get_fname_cls(f1,f2,jk_region=jk_region),f1.name,f2.name)
    except:
        print("  Computing Cl")
        cls=Spectrum.from_fields(f1,f2,bpw,wsp=get_mcm(f1,f2,jk_region=jk_region),
                                 save_windows=save_windows)
        cls.to_file(get_fname_cls(f1,f2,jk_region=jk_region))
    return cls

#gg power spectra
cls_gg={}
for fg in fields_ng:
    cls_gg[fg.name]=get_power_spectrum(fg,fg)
#yg power spectra
cls_gy={};
for fy in fields_sz:
    cls_gy[fy.name]={}
    for fg in fields_ng:
        cls_gy[fy.name][fg.name]=get_power_spectrum(fy,fg)
#yy power spectrum
cls_yy={};
for fy in fields_sz:
    cls_yy[fy.name]=get_power_spectrum(fy,fy)

#Generate model power spectra to compute the Gaussian covariance matrix
print("Generating theory power spectra")
def interpolate_spectra(leff,cell,ns):
    #Create a power spectrum interpolated at all ells
    larr=np.arange(3*ns)
    clf=interp1d(leff,cell,bounds_error=False,fill_value=0)
    clo=clf(larr)
    clo[larr<=leff[0]]=cell[0]
    clo[larr>=leff[-1]]=cell[-1]
    return clo
larr_full=np.arange(3*nside)
cls_cov_gg_data={};
cls_cov_gy_data={f.name:{} for f in fields_sz}
cls_cov_gg_model={};
cls_cov_gy_model={f.name:{} for f in fields_sz}
prof_y=Arnaud()
for fg in fields_ng:
    print(" "+fg.name)
    #Interpolate data
    larr=cls_gg[fg.name].leff
    clarr_gy={fy.name:cls_gy[fy.name][fg.name].cell for fy in fields_sz}
    cls_cov_gg_data[fg.name]=interpolate_spectra(cls_gg[fg.name].leff,
                                                 cls_gg[fg.name].cell,nside)
    for fy in fields_sz:
        cls_cov_gy_data[fy.name][fg.name]=interpolate_spectra(cls_gy[fy.name][fg.name].leff,
                                                              cls_gy[fy.name][fg.name].cell,nside)

    #Compute with model
    larr=np.arange(3*nside)
    nlarr=np.mean(cls_gg[fg.name].nell)*np.ones_like(larr)
    try:
        d=np.load(p['global']['output_dir']+'/cl_th_'+fg.name+'.npz')
        clgg=d['clgg']
        clgy=d['clgy']
    except:
        prof_g=HOD(nz_file=fg.dndz)
        clgg=hm_ang_power_spectrum(cosmo,larr,(prof_g,prof_g),
                                   zrange=fg.zrange,zpoints=64,zlog=True,
                                   **(models[fg.name]))*beam_hpix(larr,nside)**2
        clgy=hm_ang_power_spectrum(cosmo,larr,(prof_g,prof_y),
                                   zrange=fg.zrange,zpoints=64,zlog=True,
                                   **(models[fg.name]))*beam_y_planck(larr)*beam_hpix(larr,nside)**2
        np.savez(p['global']['output_dir']+'/cl_th_'+fg.name+'.npz',
                 clgg=clgg,clgy=clgy,ls=larr)
    clgg+=nlarr
    cls_cov_gg_model[fg.name]=clgg
    for fy in fields_sz:
        cls_cov_gy_model[fy.name][fg.name]=clgy
cls_cov_yy={}
for fy in fields_sz:
    cls_cov_yy[fy.name]=interpolate_spectra(cls_yy[fy.name].leff,cls_yy[fy.name].cell,nside)

#Generate covariances
print("Computing covariances")
def get_fname_cmcm(f1,f2,f3,f4):
    fname=p['global']['output_dir']+"/cmcm_"
    fname+=f1.mask_id+"_"
    fname+=f2.mask_id+"_"
    fname+=f3.mask_id+"_"
    fname+=f4.mask_id+".cmcm"
    return fname

def get_fname_cov(f1,f2,f3,f4,suffix):
    fname=p['global']['output_dir']+"/cov_"+suffix+"_"
    fname+=f1.name+"_"
    fname+=f2.name+"_"
    fname+=f3.name+"_"
    fname+=f4.name+".npz"
    return fname

def get_cmcm(f1,f2,f3,f4):
    fname=get_fname_cmcm(f1,f2,f3,f4)
    cmcm=nmt.NmtCovarianceWorkspace()
    try:
        cmcm.read_from(fname)
    except:
        print("  Computing CMCM")
        cmcm.compute_coupling_coefficients(f1.field,f2.field,f3.field,f4.field)
        cmcm.write_to(fname)
    return cmcm
        
def get_covariance(fa1,fa2,fb1,fb2,suffix,
                   cla1b1,cla1b2,cla2b1,cla2b2):
    print(" "+fa1.name+","+fa2.name+","+fb1.name+","+fb2.name)
    fname_cov=get_fname_cov(fa1,fa2,fb1,fb2,suffix)
    try:
        cov=Covariance.from_file(fname_cov,fa1.name,fa2.name,fb1.name,fb2.name)
    except:
        print("  Computing Cov")
        mcm_a=get_mcm(fa1,fa2)
        mcm_b=get_mcm(fb1,fb2)
        cmcm=get_cmcm(fa1,fa2,fb1,fb2)
        cov=Covariance.from_fields(fa1,fa2,fb1,fb2,mcm_a,mcm_b,
                                   cla1b1,cla1b2,cla2b1,cla2b2,
                                   cwsp=cmcm)
        cov.to_file(fname_cov)
    return cov

#gggg
covs_gggg_data={};
covs_gggg_model={};
dcov_gggg={}
for fg in fields_ng:
    covs_gggg_model[fg.name]=get_covariance(fg,fg,fg,fg,'model',
                                            cls_cov_gg_model[fg.name],cls_cov_gg_model[fg.name],
                                            cls_cov_gg_model[fg.name],cls_cov_gg_model[fg.name])
    covs_gggg_data[fg.name]=get_covariance(fg,fg,fg,fg,'data',
                                           cls_cov_gg_data[fg.name],cls_cov_gg_data[fg.name],
                                           cls_cov_gg_data[fg.name],cls_cov_gg_data[fg.name])
    fsky=np.mean(fg.mask)
    prof_g=HOD(nz_file=fg.dndz)
    dcov=hm_ang_1h_covariance(cosmo,fsky,cls_gg[fg.name].leff,
                              (prof_g,prof_g),(prof_g,prof_g),
                              zrange_a=fg.zrange,zpoints_a=64,zlog_a=True,
                              zrange_b=fg.zrange,zpoints_b=64,zlog_b=True,**(models[fg.name]))
    dcov_gggg[fg.name]=Covariance(fg.name,fg.name,fg.name,fg.name,dcov)
    
#gggy
covs_gggy_data={};
covs_gggy_model={};
dcov_gggy={}
for fy in fields_sz:
    covs_gggy_model[fy.name]={}
    covs_gggy_data[fy.name]={}
    dcov_gggy[fy.name]={}
    for fg in fields_ng:
        covs_gggy_model[fy.name][fg.name]=get_covariance(fg,fg,fg,fy,'model',
                                                         cls_cov_gg_model[fg.name],
                                                         cls_cov_gy_model[fy.name][fg.name],
                                                         cls_cov_gg_model[fg.name],
                                                         cls_cov_gy_model[fy.name][fg.name])
        covs_gggy_data[fy.name][fg.name]=get_covariance(fg,fg,fg,fy,'data',
                                                        cls_cov_gg_data[fg.name],
                                                        cls_cov_gy_data[fy.name][fg.name],
                                                        cls_cov_gg_data[fg.name],
                                                        cls_cov_gy_data[fy.name][fg.name])
        fsky=np.mean(fg.mask*fy.mask)
        prof_g=HOD(nz_file=fg.dndz)
        dcov=hm_ang_1h_covariance(cosmo,fsky,cls_gg[fg.name].leff,
                                  (prof_g,prof_g),(prof_g,prof_y),
                                  zrange_a=fg.zrange,zpoints_a=64,zlog_a=True,
                                  zrange_b=fg.zrange,zpoints_b=64,zlog_b=True,**(models[fg.name]))
        b_hp=beam_hpix(cls_gg[fg.name].leff,nside)
        b_y=beam_y_planck(cls_gg[fg.name].leff)
        dcov*=(b_hp**2)[:,None]*(b_hp**2*b_y)[None,:]
        dcov_gggy[fy.name][fg.name]=Covariance(fg.name,fg.name,fg.name,fy.name,dcov)
#gygy
covs_gygy_data={};
covs_gygy_model={};
dcov_gygy={}
for fy in fields_sz:
    covs_gygy_model[fy.name]={}
    covs_gygy_data[fy.name]={}
    dcov_gygy[fy.name]={}
    for fg in fields_ng:
        covs_gygy_model[fy.name][fg.name]=get_covariance(fg,fy,fg,fy,'model',
                                                         cls_cov_gg_model[fg.name],
                                                         cls_cov_gy_model[fy.name][fg.name],
                                                         cls_cov_gy_model[fy.name][fg.name],
                                                         cls_cov_yy[fy.name])
        covs_gygy_data[fy.name][fg.name]=get_covariance(fg,fy,fg,fy,'data',
                                                        cls_cov_gg_data[fg.name],
                                                        cls_cov_gy_data[fy.name][fg.name],
                                                        cls_cov_gy_data[fy.name][fg.name],
                                                        cls_cov_yy[fy.name])
        fsky=np.mean(fg.mask*fy.mask)
        prof_g=HOD(nz_file=fg.dndz)
        dcov=hm_ang_1h_covariance(cosmo,fsky,cls_gg[fg.name].leff,
                                  (prof_g,prof_y),(prof_g,prof_y),
                                  zrange_a=fg.zrange,zpoints_a=64,zlog_a=True,
                                  zrange_b=fg.zrange,zpoints_b=64,zlog_b=True,**(models[fg.name]))
        b_hp=beam_hpix(cls_gg[fg.name].leff,nside)
        b_y=beam_y_planck(cls_gg[fg.name].leff)
        dcov*=(b_hp**2*b_y)[:,None]*(b_hp**2*b_y)[None,:]
        dcov_gygy[fy.name][fg.name]=Covariance(fg.name,fy.name,fg.name,fy.name,dcov)

#Save 1-halo covariance
for fg in fields_ng:
    dcov_gggg[fg.name].to_file(p['global']['output_dir']+"/dcov_1h4pt_"+
                               fg.name+"_"+fg.name+"_"+fg.name+"_"+fg.name+".npz")
    for fy in fields_sz:
        dcov_gggy[fy.name][fg.name].to_file(p['global']['output_dir']+"/dcov_1h4pt_"+
                                            fg.name+"_"+fg.name+"_"+fg.name+"_"+fy.name+".npz")
        dcov_gygy[fy.name][fg.name].to_file(p['global']['output_dir']+"/dcov_1h4pt_"+
                                            fg.name+"_"+fy.name+"_"+fg.name+"_"+fy.name+".npz")

#Do jackknife
if p['jk']['do']:
    for jk_id in range(jk.npatches):
        if os.path.isfile(get_fname_cls(fields_sz[-1],fields_sz[-1],jk_region=jk_id)):
            print("Found %d"%(jk_id+1))
            continue
        print("%d-th JK sample out of %d"%(jk_id+1,jk.npatches))
        msk=jk.get_jk_mask(jk_id)
        #Update field
        for fg in fields_ng:
            print(" "+fg.name)
            fg.update_field(msk)
        for fy in fields_sz:
            print(" "+fy.name)
            fy.update_field(msk)

        #Compute spectra
        #gg
        for fg in fields_ng:
            get_power_spectrum(fg,fg,jk_region=jk_id,save_windows=False)
        #gy
        for fy in fields_sz:
            for fg in fields_ng:
                get_power_spectrum(fy,fg,jk_region=jk_id,save_windows=False)
        #yy
        for fy in fields_sz:
            get_power_spectrum(fy,fy,jk_region=jk_id,save_windows=False)

        #Cleanup MCMs
        if not p['jk']['store_mcm']:
            os.system("rm "+p['global']['output_dir']+'/mcm_*_jk%d.mcm'%jk_id)

    #Get covariances
    #gggg
    for fg in fields_ng:
        fname_out=get_fname_cov(fg,fg,fg,fg,"jk")
        try:
            cov=Covariance.from_file(fname_out,fg.name,fg.name,fg.name,fg.name)
        except:
            prefix=p['global']['output_dir']+'/cls_'
            prefix1=prefix+fg.name+"_"+fg.name+"_jk"
            prefix2=prefix+fg.name+"_"+fg.name+"_jk"
            cov=Covariance.from_jk(jk.npatches,prefix1,prefix2,".npz",
                                   fg.name,fg.name,fg.name,fg.name)
        cov.to_file(fname_out,n_samples=jk.npatches)

    for fy in fields_sz:
        for fg in fields_ng:
            #gggy
            fname_out=get_fname_cov(fg,fg,fg,fy,"jk")
            try:
                cov=Covariance.from_file(fname_out,fg.name,fg.name,fg.name,fy.name)
            except:
                prefix=p['global']['output_dir']+'/cls_'
                prefix1=prefix+fg.name+"_"+fg.name+"_jk"
                prefix2=prefix+fy.name+"_"+fg.name+"_jk"
                cov=Covariance.from_jk(jk.npatches,prefix1,prefix2,".npz",
                                       fg.name,fg.name,fg.name,fy.name)
            cov.to_file(fname_out,n_samples=jk.npatches)

            #gygy
            fname_out=get_fname_cov(fg,fy,fg,fy,"jk")
            try:
                cov=Covariance.from_file(fname_out,fg.name,fy.name,fg.name,fy.name)
            except:
                prefix=p['global']['output_dir']+'/cls_'
                prefix1=prefix+fy.name+"_"+fg.name+"_jk"
                prefix2=prefix+fy.name+"_"+fg.name+"_jk"
                cov=Covariance.from_jk(jk.npatches,prefix1,prefix2,".npz",
                                       fg.name,fy.name,fg.name,fy.name)
            cov.to_file(fname_out,n_samples=jk.npatches)
