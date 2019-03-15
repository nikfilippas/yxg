import numpy as np
import pyccl as ccl
from pspec import ang_power_spectrum, power_spectrum
from profile2D import Arnaud,HOD
from scipy.integrate import simps
import matplotlib.pyplot as plt
import healpy as hp
from covariance import ang_1halo_covariance
import sys

if len(sys.argv)!=3:
    raise ValueError("Usage: get_covariance.py sample_name bin_number")

g_sample=sys.argv[1]
bin_no=int(sys.argv[2])
y_sample='y_milca'
prefix_cls='../analysis/out_ns512_linlog/'
prefix_nz='../analysis/data/dndz/'
kmax=1.

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
fsky=np.mean(hp.read_map("../analysis/data/maps/mask_v3.fits",verbose=False));

#Scale cuts
zmean=np.sum(z*nz)/np.sum(nz)
chimean=ccl.comoving_radial_distance(cosmo,1/(1.+zmean))
lmax=int(kmax*chimean+0.5)
mask_gg=d_gg['leff']<lmax
mask_gy=d_gy['leff']<lmax

#Form data vector and covariance
nl_gg=len(mask_gg)
nl_gy=len(mask_gy)
ls_gg=d_gg['leff']
ls_gy=d_gy['leff']
beam_gg=np.ones(nl_gg)*beam_hpix(ls_gg)**2
beam_gy=beam_ymap(ls_gy)*beam_hpix(ls_gy)**2
dv_gg=(d_gg['cell']-d_gg['nell'])
dv_gy=(d_gy['cell']-d_gy['nell'])
cv_gg_gg_G=d_gggg['cov']
cv_gg_gy_G=d_gggy['cov']
cv_gy_gy_G=d_gygy['cov']
cv_gy_gg_G=cv_gg_gy_G.T

#Read best-fit parameters
res=np.load('result_'+g_sample+ibin_string+'_11.npz')
params=dict(zip(res['names'],res['values']))

#Set up profiles
py=Arnaud(y_sample)
pg=HOD(g_sample+ibin_string,nz_file=fname_dndz)

dcv_gg_gg_NG=ang_1halo_covariance(cosmo,fsky,ls_gg,(pg,pg),(pg,pg),
                                  zrange_a=z_range,zpoints_a=128,zlog_a=True,
                                  zrange_b=z_range,zpoints_b=128,zlog_b=True,**params)
dcv_gg_gy_NG=ang_1halo_covariance(cosmo,fsky,ls_gg,(pg,pg),(pg,py),
                                  zrange_a=z_range,zpoints_a=128,zlog_a=True,
                                  zrange_b=z_range,zpoints_b=128,zlog_b=True,**params)
dcv_gy_gy_NG=ang_1halo_covariance(cosmo,fsky,ls_gg,(pg,py),(pg,py),
                                  zrange_a=z_range,zpoints_a=128,zlog_a=True,
                                  zrange_b=z_range,zpoints_b=128,zlog_b=True,**params)
dcv_gy_gg_NG=dcv_gg_gy_NG.T

tv_gg=ang_power_spectrum(cosmo,ls_gg,(pg,pg),zrange=z_range,zpoints=128,zlog=True,**params)*beam_gg
tv_gy=ang_power_spectrum(cosmo,ls_gy,(pg,py),zrange=z_range,zpoints=128,zlog=True,**params)*beam_gy

cv_gg_gg_NG=cv_gg_gg_G+dcv_gg_gg_NG
cv_gg_gy_NG=cv_gg_gy_G+dcv_gy_gg_NG
cv_gy_gg_NG=cv_gy_gg_G+dcv_gg_gy_NG
cv_gy_gy_NG=cv_gy_gy_G+dcv_gy_gy_NG

plt.figure()
plt.plot(ls_gg,np.diag(cv_gg_gg_G),'r--')
plt.plot(ls_gg,np.diag(cv_gg_gg_NG),'k-')
plt.loglog()

plt.figure()
plt.plot(ls_gy,np.diag(cv_gy_gy_G),'r--')
plt.plot(ls_gy,np.diag(cv_gy_gy_NG),'k-')
plt.loglog()

plt.figure()
plt.plot(ls_gy,np.diag(cv_gg_gy_G),'r--')
plt.plot(ls_gy,np.diag(cv_gg_gy_NG),'k-')
plt.loglog()

cv_tot_G=np.zeros([nl_gg+nl_gy,nl_gg+nl_gy])
cv_tot_NG=np.zeros([nl_gg+nl_gy,nl_gg+nl_gy])
cv_tot_G[:nl_gg,:][:,:nl_gg]=cv_gg_gg_G
cv_tot_G[:nl_gg,:][:,nl_gg:]=cv_gg_gy_G
cv_tot_G[nl_gg:,:][:,:nl_gg]=cv_gy_gg_G
cv_tot_G[nl_gg:,:][:,nl_gg:]=cv_gy_gy_G
cv_tot_NG[:nl_gg,:][:,:nl_gg]=cv_gg_gg_NG
cv_tot_NG[:nl_gg,:][:,nl_gg:]=cv_gg_gy_NG
cv_tot_NG[nl_gg:,:][:,:nl_gg]=cv_gy_gg_NG
cv_tot_NG[nl_gg:,:][:,nl_gg:]=cv_gy_gy_NG

plt.figure()
plt.imshow(cv_tot_G/np.sqrt(np.diag(cv_tot_G)[:,None]*np.diag(cv_tot_G)[None,:]))

plt.figure()
plt.imshow(cv_tot_NG/np.sqrt(np.diag(cv_tot_NG)[:,None]*np.diag(cv_tot_NG)[None,:]))

plt.figure()
plt.errorbar(ls_gg,dv_gg,yerr=np.sqrt(np.diag(cv_gg_gg_NG)),fmt='r.')
plt.plot(ls_gg,tv_gg,'k-')
plt.xlim([0.9*ls_gg[0],lmax])
plt.loglog()

plt.figure()
plt.errorbar(ls_gy,dv_gy,yerr=np.sqrt(np.diag(cv_gy_gy_NG)),fmt='r.')
plt.plot(ls_gy,tv_gy,'k-')
plt.xlim([0.9*ls_gy[0],lmax])
plt.loglog()
plt.show()

np.savez('dcov_'+g_sample+ibin_string+'_'+g_sample+ibin_string+'_'+g_sample+ibin_string+'_'+g_sample+ibin_string,
         cov=dcv_gg_gg_NG)
np.savez('dcov_'+g_sample+ibin_string+'_'+g_sample+ibin_string+'_'+g_sample+ibin_string+'_'+y_sample,
         cov=dcv_gg_gy_NG)
np.savez('dcov_'+g_sample+ibin_string+'_'+y_sample+'_'+g_sample+ibin_string+'_'+y_sample,
         cov=dcv_gy_gy_NG)

def get_chi2(d,t,c,m):
    dx=(d-t)[m]
    cv=c[m,:][:,m]
    icv=np.linalg.inv(cv)
    print(np.einsum('i,ij,j',dx,icv,dx),len(dx))
get_chi2(dv_gy,tv_gy,cv_gy_gy_G,mask_gy)
get_chi2(dv_gy,tv_gy,cv_gy_gy_NG,mask_gy)
