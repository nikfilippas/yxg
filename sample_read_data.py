import numpy as np
import matplotlib.pyplot as plt

#Read power spectrum data
dcl=np.load("cl_2mpz_2mpz.npz")
#Read covariance data
dcov=np.load("cov_2mpz_2mpz_2mpz_2mpz.npz")

#This is the ells at which the power spectrm is sampled
ells=dcl['leff']
#This is the signal+noise power spectrum
cells_with_noise=dcl['cell']
#This is the noise power spectrum
nells=dcl['nell']
#To obtain the power spectrum that you need to put into the likelihood, you need to subtract them
cells=cells_with_noise-nells
#This is the covariance matrix
covar=dcov['cov']

#Now, you will want to cut the power spectrum because at very high ells, our models are not accurate enough.
#For 2MPZ this means cutting at ell>~260. So let's do that first
mask=ells<260
ells=ells[mask]
cells=cells[mask]
#You also need to cut the covariance matrix!
covar=covar[mask,:][:,mask]

#Now let's plot the results
plt.figure()
#Note that the error bars are given by the diagonal of the covariance matrix
err_ell=np.sqrt(np.diag(covar))
plt.errorbar(ells,cells,yerr=err_ell,fmt='r.')
plt.loglog()
plt.xlabel('$\\ell$',fontsize=15)
plt.ylabel('$C_\\ell$',fontsize=15)
plt.show()
