from astropy.io import fits
from healpy.visufunc import mollview
from matplotlib import cm

dir1 = "../../data/maps/"

f1 = dir1 + "mask_planck60.fits"
f2 = dir1 + "mask_planck60S.fits"

data1 = fits.getdata(f1, ext=0)
data2 = fits.getdata(f2, ext=0)


mollview(data1, title="", cbar=False, cmap=cm.Greys_r)
mollview(data2, title="", cbar=False, cmap=cm.Greys_r)
