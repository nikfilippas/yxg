"""
"""

import numpy as np
import pyccl as ccl
from hod import HODProfile
import matplotlib.pyplot as plt
from matplotlib import cm

lmminf=lambda x: 12.
sigmf=lambda x: 0.5
m0f=lambda x:1.584893E+12
m1f=lambda x:4.466836E+13
alphaf=lambda x:1.0
fcf=lambda x:0.8

##############################################################
#                                                            #
# This script illustrates how to generate predictions        #
# for the angular power spectrum of HOD models.              #
# Note that this only works with an unofficial branch        #
# of CCL: https://github.com/LSSTDESC/CCL/tree/lss_hsc_work  #
#                                                            #
##############################################################

#Initialize HOD profile
hod=HODProfile(lmminf,sigmf,fcf,m0f,m1f,alphaf,Delta=500,is_delta_matter=False)

#Initialize CCL cosmology
cosmo=ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96)
karr=np.logspace(-4.,2.,512)
zarr=np.linspace(0.,3.,64)[::-1]

#Compute power spectrum at a given redshift (just for illustrative purposes)
pkarr,p1harr,p2harr,nkarr,bkarr=hod.pk(cosmo,0.5,karr,return_decomposed=True)
#Plot for fun
plt.figure()
plt.plot(karr,p1harr,'r-',label='1-halo')
plt.plot(karr,p2harr,'b-',label='2-halo')
plt.plot(karr,pkarr,'k-',label='Total')
plt.plot(karr,nkarr,'k--',lw=1,label='Shot noise')
plt.legend(loc='lower left')
plt.xlim([1E-4,1E2])
plt.loglog()
plt.xlabel('$k\\,\\,[{\\rm Mpc}^{-1}]$',fontsize=15)
plt.ylabel('$P(k)\\,\\,[{\\rm Mpc}^{-1}]$',fontsize=15)
plt.show()
