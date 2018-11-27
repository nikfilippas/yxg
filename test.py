import numpy as np
import pyccl as ccl
import profile2D
import matplotlib.pyplot as plt
import pspec

cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.66, sigma8=0.79, n_s=0.81)

pa=profile2D.Arnaud(rrange=(1E-3,1E3),qpoints=1024)

larr=np.logspace(1.,np.log10(2E4),128)
cl_1h=pspec.ang_power_spectrum(cosmo,larr,pa,pa,include_2h=False)
cl_2h=pspec.ang_power_spectrum(cosmo,larr,pa,pa,include_1h=False)
cl_tt=cl_1h+cl_2h

plt.plot(larr,cl_1h*1E12*larr*(larr+1)/(2*np.pi),'r-',label='1-halo')
plt.plot(larr,cl_2h*1E12*larr*(larr+1)/(2*np.pi),'b--',label='2-halo')
plt.plot(larr,cl_tt*1E12*larr*(larr+1)/(2*np.pi),'y-.',label='Total')
plt.xlabel('$\\ell$',fontsize=15)
plt.ylabel('$10^{12}\\ell(\\ell+1)\\,C_\\ell/2\\pi$',fontsize=15)
plt.ylim([5E-3,2])
plt.loglog()
plt.legend(loc='lower right')
plt.show()
