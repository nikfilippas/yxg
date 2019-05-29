import healpy as hp
from astropy.io import fits
import numpy as np
import pyccl as ccl

RHOCRIT0=2.7744948E11
HH=0.67
cosmo=ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=HH, sigma8=0.8, n_s=0.96)
nside=2048

print(" Making SZ point source mask")
def rDelta(m,zz,Delta) :
    """Returns r_Delta
    """
    hn=ccl.h_over_h0(cosmo,1./(1+zz))
    rhoc=RHOCRIT0*hn*hn
    return (3*m/(4*np.pi*Delta*rhoc))**0.333333333*(1+zz)

#Read catalog and remove all clusters above z=0.43
data=(fits.open('data/maps/HFI_PCCS_SZ-union_R2.08.fits'))[1].data
mask = (data['REDSHIFT']>=0) & (data['SNR']>=6)
data=data[mask]
#Compute their angular extent
r500=rDelta(data['MSZ']*HH*1E14,data['REDSHIFT'],500)
chi=ccl.comoving_radial_distance(cosmo,1./(1+data['REDSHIFT']))*HH
th500=r500/chi
#Compute angular positions for each cluster
theta=(90-data['GLAT'])*np.pi/180
phi=data['GLON']*np.pi/180
vx=np.sin(theta)*np.cos(phi)
vy=np.sin(theta)*np.sin(phi)
vz=np.cos(theta)
#Generate mask by cutting out a circle of radius
#3*theta_500 around each cluster
mask_sz=np.ones(hp.nside2npix(nside))
for i in np.arange(len(data)) :
    v=np.array([vx[i],vy[i],vz[i]])
    radius=3*th500[i]
    ip=hp.query_disc(nside,v,radius)
    mask_sz[ip]=0
hp.write_map("data/maps/mask_sz.fits",mask_sz,overwrite=True)

print("Reading official Planck masks")
mask_gal_80=hp.read_map("data/maps/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",verbose=False,field=4)
mask_gal_60=hp.read_map("data/maps/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",verbose=False,field=2)
mask_gal_40=hp.read_map("data/maps/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",verbose=False,field=1)
mask_gal_20=hp.read_map("data/maps/HFI_Mask_GalPlane-apo0_2048_R2.00.fits",verbose=False,field=0)
mask_p0=hp.read_map("data/maps/LFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,hdu=1);
mask_p1=hp.read_map("data/maps/LFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,hdu=2);
mask_p2=hp.read_map("data/maps/LFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,hdu=3);
mask_pl=mask_p0*mask_p1*mask_p2
mask_p0=hp.read_map("data/maps/HFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,field=0);
mask_p1=hp.read_map("data/maps/HFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,field=1);
mask_p2=hp.read_map("data/maps/HFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,field=2);
mask_p3=hp.read_map("data/maps/HFI_Mask_PointSrc_2048_R2.00.fits",verbose=False,field=3);
mask_ph=mask_p0*mask_p1*mask_p2*mask_p3

print("Writing output masks")
hp.write_map("data/maps/mask_planck20.fits",mask_gal_20*mask_ph,overwrite=True)
hp.write_map("data/maps/mask_planck40.fits",mask_gal_40*mask_ph,overwrite=True)
hp.write_map("data/maps/mask_planck60.fits",mask_gal_60*mask_ph,overwrite=True)
hp.write_map("data/maps/mask_planck80.fits",mask_gal_80*mask_ph,overwrite=True)
hp.write_map("data/maps/mask_planck60L.fits",mask_gal_60*mask_ph*mask_pl,overwrite=True)
hp.write_map("data/maps/mask_planck80L.fits",mask_gal_80*mask_ph*mask_pl,overwrite=True)
hp.write_map("data/maps/mask_planck20S.fits",mask_gal_20*mask_ph*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck40S.fits",mask_gal_40*mask_ph*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck60S.fits",mask_gal_60*mask_ph*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck80S.fits",mask_gal_80*mask_ph*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck20LS.fits",mask_gal_20*mask_ph*mask_pl*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck40LS.fits",mask_gal_40*mask_ph*mask_pl*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck60LS.fits",mask_gal_60*mask_ph*mask_pl*mask_sz,overwrite=True)
hp.write_map("data/maps/mask_planck80LS.fits",mask_gal_80*mask_ph*mask_pl*mask_sz,overwrite=True)
