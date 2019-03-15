import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
from profile2D import Arnaud,HOD
from pspec import ang_power_spectrum
from scipy.optimize import minimize
import sys

if len(sys.argv)!=4:
    raise ValueError("Usage: MCMC_dam.py sample_name bin_number fit_gy")

prefix_cls='../analysis/out_ns512_linlog/'
prefix_nz='../analysis/data/dndz/'
g_sample=sys.argv[1]
bin_no=int(sys.argv[2])
y_sample='y_milca'
kmax=1.
fit_gg=True
fit_gy=bool(int(sys.argv[3]))

#y-map beam
nside=512
fwhm_y_amin=10.
fwhm_hp_amin=60*41.7/nside
sigma_y_rad=np.radians(fwhm_y_amin/2.355/60)
sigma_hp_rad=np.radians(fwhm_hp_amin/2.355/60)
def beam_ymap(l):
    return np.exp(-0.5*l*(l+1)*sigma_y_rad**2)
def beam_hpix(l):
    return np.exp(-0.5*l*(l+1)*sigma_hp_rad**2)

#Fiducial cosmology
cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665)

#Read data
fname_dndz=prefix_nz+g_sample.upper()
if bin_no<0:
    ibin_string=''
    fname_dndz+='_bin1.txt'
else:
    ibin_string='_b%d'%bin_no
    fname_dndz+='_bin%d.txt'%bin_no
fname_cl_gg=prefix_cls+'cl_'+g_sample+ibin_string+'_'+g_sample+ibin_string+'.npz'
fname_cl_gy=prefix_cls+'cl_'+g_sample+ibin_string+'_'+y_sample+'.npz'
fname_cov_gg_gg=prefix_cls+'cov_'+g_sample+ibin_string+'_'+g_sample+ibin_string+'_'+g_sample+ibin_string+'_'+g_sample+ibin_string+'.npz'
fname_cov_gg_gy=prefix_cls+'cov_'+g_sample+ibin_string+'_'+g_sample+ibin_string+'_'+g_sample+ibin_string+'_'+y_sample+'.npz'
fname_cov_gy_gy=prefix_cls+'cov_'+g_sample+ibin_string+'_'+y_sample+'_'+g_sample+ibin_string+'_'+y_sample+'.npz'
z,nz=np.loadtxt(fname_dndz,unpack=True)
z_inrange=z[(nz>0.005*np.amax(nz))];
z_range=np.array([z_inrange[0],z_inrange[-1]])
d_gg=np.load(fname_cl_gg)
d_gy=np.load(fname_cl_gy)
d_gggg=np.load(fname_cov_gg_gg)
d_gggy=np.load(fname_cov_gg_gy)
d_gygy=np.load(fname_cov_gy_gy)

#Scale cuts
zmean=np.sum(z*nz)/np.sum(nz)
chimean=ccl.comoving_radial_distance(cosmo,1/(1.+zmean))
lmax=int(kmax*chimean+0.5)
mask_gg=d_gg['leff']<lmax
mask_gy=d_gy['leff']<lmax

#Form data vector and covariance
nl_gg=np.sum(mask_gg)
nl_gy=np.sum(mask_gy)
ls_gg=d_gg['leff'][mask_gg]
ls_gy=d_gy['leff'][mask_gy]
beam_gg=np.ones(nl_gg)*beam_hpix(ls_gg)**2
beam_gy=beam_ymap(ls_gy)*beam_hpix(ls_gy)**2
dv_gg=(d_gg['cell']-d_gg['nell'])[mask_gg]
dv_gy=(d_gy['cell']-d_gy['nell'])[mask_gy]
cv_gg_gg=d_gggg['cov'][mask_gg,:][:,mask_gg]
cv_gg_gy=d_gggy['cov'][mask_gg,:][:,mask_gy]
cv_gy_gy=d_gygy['cov'][mask_gy,:][:,mask_gy]
cv_gy_gg=cv_gg_gy.T
dv_tot=np.concatenate((dv_gg,dv_gy))
ls_tot=np.concatenate((ls_gg,ls_gy))
beam_tot=np.concatenate((beam_gg,beam_gy))
cv_tot=np.zeros([nl_gg+nl_gy,nl_gg+nl_gy])
cv_tot[:nl_gg,:][:,:nl_gg]=cv_gg_gg
cv_tot[:nl_gg,:][:,nl_gg:]=cv_gg_gy
cv_tot[nl_gg:,:][:,:nl_gg]=cv_gy_gg
cv_tot[nl_gg:,:][:,nl_gg:]=cv_gy_gy
if fit_gg:
    if fit_gy:
        dv=dv_tot
        cv=cv_tot
        ls=ls_tot
        beam=beam_tot
    else:
        dv=dv_gg
        cv=cv_gg_gg
        ls=ls_gg
        beam=beam_gg
elif fit_gy:
    dv=dv_gy
    cv=cv_gy
    ls=ls_gy
    beam=beam_gy
else:
    raise ValueError("Must fit either gg or gy")
icv=np.linalg.inv(cv)

#Set up profiles
prof_y=Arnaud(y_sample)
prof_g=HOD(g_sample+ibin_string,nz_file=fname_dndz)

#Define likelihood
priors={'fc':[1.,1.,1.],
        'Mmin':[12.,10.,16.],
        "M1":[13.5,10.,16.],
        "M0":"Mmin",
        "alpha":[1.,1.,1.],
        "beta_max":[1.,1.,1.],
        "beta_gal":[1.,1.,1.],
        "sigma_lnM":[0.15,0.15,0.15],
        "b_hydro":[0.3,0.,1.0],
        "r_corr":[0,-1.,1.]
        }
if not fit_gy: #Fix b_hydro if not fitting gy
    priors['b_hydro'][1]=priors['b_hydro'][2]=priors['b_hydro'][0]
params_free_names=[]
params_free_priors=[]
params_fixed=[]
params_coupled=[]
p0=[]
for k in sorted(priors.keys()):
    if type(priors[k])==str:
        params_coupled.append((k,priors[k]))
    elif priors[k][1]==priors[k][2]:
        params_fixed.append((k,priors[k][0]))
    else:
        params_free_names.append(k)
        params_free_priors.append(np.array(priors[k])[1:])
        p0.append(priors[k][0])

def build_kwargs(p):
    params=dict(params_fixed)
    params.update(dict(zip(params_free_names,p)))
    for p1,p2 in params_coupled:
        params[p1]=params[p2]
    return params

def lnprior(p):
    if any([not(pr[0]<= pp <= pr[1]) for pp,pr in zip(p,params_free_priors)]):
        return -np.inf
    return 0

def get_theory(z_log_gg=True,z_points_gg=128,z_log_gy=True,z_points_gy=128,**kwargs):
    tv_gg=[]; tv_gy=[];
    if fit_gg:
        tv_gg=ang_power_spectrum(cosmo,ls_gg,(prof_g,prof_g),zrange=z_range,zpoints=z_points_gg,zlog=z_log_gg,**kwargs)
    if fit_gy:
        tv_gy=ang_power_spectrum(cosmo,ls_gy,(prof_g,prof_y),zrange=z_range,zpoints=z_points_gy,zlog=z_log_gy,**kwargs)
    if (tv_gg is None) or (tv_gy is None):
        return None
    return np.concatenate((tv_gg,tv_gy))*beam
    
def lnlike(p):
    params=build_kwargs(p)
    tv=get_theory(**params)
    if tv is None:
        return -np.inf
    dx=dv-tv
    print(params)
    return -0.5*np.einsum('i,ij,j',dx,icv,dx)

def plot_theory(p):
    params=build_kwargs(p)
    tv=get_theory(**params)
    if fit_gg and fit_gy:
        tv_gg=tv[:nl_gg]
        tv_gy=tv[nl_gg:]
        plt.errorbar(ls_gg,dv_gg,yerr=np.sqrt(np.diag(cv_gg_gg)),fmt='r.')
        plt.errorbar(ls_gy,dv_gy,yerr=np.sqrt(np.diag(cv_gy_gy)),fmt='b.')
        plt.plot(ls_gg,tv_gg,'r-',label='gg')
        plt.plot(ls_gy,tv_gy,'b-',label='yy')
    else:
        plt.errorbar(ls,dv,yerr=np.sqrt(np.diag(cv)),fmt='k.')
        plt.plot(ls,tv,'k-')
    plt.loglog()
    plt.xlabel('$\\ell$',fontsize=16)
    plt.ylabel('$C_\\ell$',fontsize=16)

def lnprob(p,sign=+1):
    pr=lnprior(p)
    if pr!=-np.inf:
        pr+=lnlike(p)
    print(pr)
    return sign*pr

print("Minimizing")
res=minimize(lnprob,p0,method='Powell',args=(-1))
print(g_sample+ibin_string,build_kwargs(res.x))
print("Computing covariance")
import numdifftools as nd
covar=np.linalg.inv(-nd.Hessian(lnprob)(res.x))
pars_final=build_kwargs(res.x)
names=sorted(pars_final.keys())
values=np.array([pars_final[k] for k in names])
np.savez("result_"+g_sample+ibin_string+"_%d%d"%(int(fit_gg),int(fit_gy)),names=names,values=values,names_var=params_free_names,covar=covar)
plot_theory(res.x)
plt.savefig("result_"+g_sample+ibin_string+"_%d%d.pdf"%(int(fit_gg),int(fit_gy)),bbox_inches='tight')
plt.show()
