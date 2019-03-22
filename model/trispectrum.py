import numpy as np
import pyccl as ccl
from scipy.integrate import simps
import matplotlib.pyplot as plt
import healpy as hp

def hm_1h_trispectrum(cosmo,k,a,profiles,logMrange=(6,17),mpoints=128,**kwargs):
    pau,pav,pbu,pbv=profiles
    
    aUnorm=pau.profnorm(cosmo,a,squeeze=False,**kwargs)
    aVnorm=pav.profnorm(cosmo,a,squeeze=False,**kwargs)
    bUnorm=pbu.profnorm(cosmo,a,squeeze=False,**kwargs)
    bVnorm=pbv.profnorm(cosmo,a,squeeze=False,**kwargs)
    
    logMmin, logMmax=logMrange
    mpoints=int(mpoints)
    M=np.logspace(logMmin,logMmax,mpoints)

    Dm=pau.Delta/ccl.omega_x(cosmo,a,'matter')
    mfunc=np.array([ccl.massfunc(cosmo,M,aa,Dmm) for aa,Dmm in zip(a,Dm)]).T

    aU,aUU=pau.fourier_profiles(cosmo,k,M,a,squeeze=False,**kwargs)
    if pau.name==pav.name:
        aUV=aUU
    else:
        aV,aVV = pav.fourier_profiles(cosmo,k,M,a,squeeze=False,**kwargs)
        if 'r_corr' in kwargs:
            r=kwargs['r_corr']
        else:
            r=0
        aUV=np.sqrt(aUU*aVV)*(1+r)

    bU,bUU=pbu.fourier_profiles(cosmo,k,M,a,squeeze=False,**kwargs)
    if pbu.name==pbv.name:
        bUV=bUU
    else:
        bV,bVV = pbv.fourier_profiles(cosmo,k,M,a,squeeze=False,**kwargs)
        if 'r_corr' in kwargs:
            r=kwargs['r_corr']
        else:
            r=0
        bUV=np.sqrt(bUU*bVV)*(1+r)

    t1h=simps(mfunc[:,:,None,None]*aUV[:,:,:,None]*bUV[:,:,None,:],x=np.log10(M),axis=0)

    rhoM = ccl.rho_x(cosmo, a, "matter", is_comoving=True)
    dlM = (logMmax-logMmin) / (mpoints-1)
    n0_1h=(rhoM-np.dot(M,mfunc)*dlM)/M[0]
    t1h+=(n0_1h[:,None,None]*aUV[0,:,:,None]*bUV[0,:,None,:])
    t1h/=(aUnorm*aVnorm*bUnorm*bVnorm)[:,None,None]

    return t1h

def hm_ang_1h_covariance(cosmo, fsky, l, profiles_a, profiles_b,
                         zrange_a=(1e-6,6), zpoints_a=32, zlog_a=True,
                         zrange_b=(1e-6,6), zpoints_b=32, zlog_b=True,
                         logMrange=(6,17), mpoints=128,**kwargs):
    
    zrange=np.array([min(np.amin(zrange_a),np.amin(zrange_b)),
                     max(np.amax(zrange_a),np.amax(zrange_b))])
    dz=min((zrange_a[1]-zrange_a[0])/zpoints_a,(zrange_b[1]-zrange_b[0])/zpoints_b)
    zpoints=int((zrange[1]-zrange[0])/dz)
    zlog=zlog_a or zlog_b

    zmin, zmax = zrange
    # Distance measures & out-of-loop optimisations
    if zlog:
        z = np.geomspace(zmin, zmax, zpoints)
        jac = z
        x= np.log(z)
    else:
        z = np.linspace(zmin, zmax, zpoints)
        jac = 1
        x = z
    a = 1/(1+z)
    chi = ccl.comoving_radial_distance(cosmo, a)
    
    H_inv = 2997.92458 * jac/(ccl.h_over_h0(cosmo, a)*cosmo["h"])  # c*z/H(z)
    pau,pav=profiles_a
    pbu,pbv=profiles_b
    aWu=pau.kernel(cosmo,a)
    aWv=pav.kernel(cosmo,a)
    bWu=pbu.kernel(cosmo,a)
    bWv=pbv.kernel(cosmo,a)
    N=H_inv*aWu*aWv*bWu*bWv/chi**6

    k=(l+1/2)/chi[...,None]
    t1h=hm_1h_trispectrum(cosmo,k,a,(pau,pav,pbu,pbv),logMrange,mpoints,**kwargs)

    tl = simps(N[:,None,None]*t1h, x, axis=0)

    return tl/(4*np.pi*fsky)
