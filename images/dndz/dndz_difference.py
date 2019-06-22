import os
os.chdir("../../")
import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
from model.power_spectrum import HalomodCorrection,hm_ang_power_spectrum
from model.profile2D import Arnaud, HOD
from model.utils import beam_gaussian,beam_hpix

# This script produces a plot comparing the theoretical predictions and data for two different N(z)s

# File names
sample_name='wisc2'
fname_nz1="data/dndz/WISC_bin2.txt"
fname_nz2="data/dndz/WISC_v2_bin2.txt"

# Read data
d_gg=np.load("output/cls_"+sample_name+"_"+sample_name+".npz")
d_gy=np.load("output/cls_y_milca_"+sample_name+".npz")
ls=d_gg['ls']
cl_gg=d_gg['cls']-d_gg['nls']
cl_gy=d_gy['cls']-d_gy['nls']
cov_gggg=np.load("output/cov_comb_m_"
                 +sample_name+"_"+sample_name+"_"
                 +sample_name+"_"+sample_name+".npz")['cov']
cov_gggy=np.load("output/cov_comb_m_"
                 +sample_name+"_"+sample_name
                 +"_"+sample_name+"_y_milca.npz")['cov']
cov_gygy=np.load("output/cov_comb_m_"
                 +sample_name+"_y_milca_"
                 +sample_name+"_y_milca.npz")['cov']
z1,dndz1=np.loadtxt(fname_nz1,unpack=True)
z2,dndz2=np.loadtxt(fname_nz2,unpack=True)

# Read best-fit parameters
pars=np.load("output/sampler_run_kmax_"+sample_name+"_properties.npz")
bf_params=dict(zip(pars['names'],pars['p0']))
bf_params['M0']=bf_params['Mmin']
bf_params['fc']=1
bf_params['alpha']=1
bf_params['beta_max']=1
bf_params['beta_gal']=1
bf_params['sigma_lnM']=0.15

# Redshift ranges
z1_inrange = z1[dndz1 >= 0.005*np.amax(dndz1)]
z1_range = [z1_inrange[0], z1_inrange[-1]]
z2_inrange = z2[dndz2 >= 0.005*np.amax(dndz2)]
z2_range = [z2_inrange[0], z2_inrange[-1]]

# Cosmology and halo model correction
cosmo=ccl.Cosmology(Omega_c=0.26066676,
                    Omega_b=0.048974682,
                    h=0.6766,
                    sigma8=0.8102,
                    n_s=0.9665,
                    mass_function='tinker')
hm_correction=HalomodCorrection(cosmo)

# Lmax
kmax=1.
zmean=np.sum(z1*dndz1)/np.sum(dndz1)
chimean=ccl.comoving_radial_distance(cosmo,1./(1+zmean))
lmax=kmax*chimean-0.5
mask=ls<=lmax
ls=ls[mask]
cl_gg=cl_gg[mask]
cl_gy=cl_gy[mask]
cov_gggg=cov_gggg[mask,:][:,mask]
cov_gggy=cov_gggy[mask,:][:,mask]
cov_gygy=cov_gygy[mask,:][:,mask]

# Beams
b_hp=beam_hpix(ls,512)
b_y=beam_gaussian(ls, 10.)

# Profiles
prof_g1=HOD("dndz1",fname_nz1)
prof_g2=HOD("dndz2",fname_nz2)
prof_y=Arnaud("y_milca")

# Cls for first redshift distribution
clt_gg1=hm_ang_power_spectrum(cosmo, ls, (prof_g1,prof_g1),
                              zrange=z1_range,zpoints=32,zlog=True,
                              hm_correction=hm_correction,selection=None,
                              **bf_params)*b_hp**2
clt_gy1=hm_ang_power_spectrum(cosmo, ls, (prof_g1,prof_y),
                              zrange=z1_range,zpoints=32,zlog=True,
                              hm_correction=hm_correction,selection=None,
                              **bf_params)*b_hp**2*b_y
# Cls for second redshift distribution
clt_gg2=hm_ang_power_spectrum(cosmo, ls, (prof_g2,prof_g2),
                              zrange=z1_range,zpoints=32,zlog=True,
                              hm_correction=hm_correction,selection=None,
                              **bf_params)*b_hp**2
clt_gy2=hm_ang_power_spectrum(cosmo, ls, (prof_g2,prof_y),
                              zrange=z1_range,zpoints=32,zlog=True,
                              hm_correction=hm_correction,selection=None,
                              **bf_params)*b_hp**2*b_y

# Plotting
def plot_data(d,c,t1,t2,title='',fname=None):
    sigma=np.sqrt(np.diag(c))
    fig=plt.figure()
    ax1 = fig.add_axes((.1,.3,.8,.6))
    ax1.set_title(title)
    ax1.errorbar(ls,d,yerr=sigma,fmt='r.',label='Data')
    ax1.plot(ls,t1,'k-',label='First N(z)')
    ax1.plot(ls,t2,'k--',label='Second N(z)')
    ax1.set_ylabel("$C_\\ell$",fontsize=15)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim([np.amin(t1)*0.9,1.2*np.amax(t1)])
    ax1.legend(loc='upper right',frameon=False)
    ax2=fig.add_axes((.1,.1,.8,.2))
    ax2.errorbar(ls, (d-t1)/sigma,yerr=np.ones_like(d),fmt='r.')
    ax2.plot(ls,(t1-t1)/sigma,'k-')
    ax2.plot(ls,(t2-t1)/sigma,'k--')
    ax2.set_xscale('log')
    ax2.set_xlabel('$\\ell$',fontsize=15)
    ax2.set_ylabel('$\\Delta C_\\ell/\\sigma_\\ell$',fontsize=15)
    if fname is not None:
        plt.savefig(fname,bbox_inches='tight')

os.chdir("images/dndz/")
plt.figure()
plt.plot(ls,clt_gg2/clt_gg1)
plt.savefig("ratio.pdf")
plot_data(cl_gg,cov_gggg,clt_gg1,clt_gg2,
          title='WIxSC - bin 2, gg',
          fname="cl_"+sample_name+"_gg_nzcomp.pdf")
plot_data(cl_gy,cov_gygy,clt_gy1,clt_gy2,
          title='WIxSC - bin 2, gy',
          fname="cl_"+sample_name+"_gy_nzcomp.pdf")
plt.show()
