import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

gsample='wisc2'
ls=np.load("output_dam/cls_y_milca_"+gsample+".npz")['ls']
c1=np.load("output_dam/cov_model_"+gsample+"_y_milca_"+gsample+"_y_milca.npz")['cov']
d1=np.load("output_dam/dcov_1h4pt_"+gsample+"_y_milca_"+gsample+"_y_milca.npz")['cov']
c2=np.load("output_dam/cov_jk_"+gsample+"_y_milca_"+gsample+"_y_milca.npz")['cov']
c1+=d1
plt.figure()
ax=plt.gca()
ax.set_title('WI$\\times$SC-2 $-\\,y_{\\rm MILCA}$',
             fontsize=15)
ax.plot(ls,np.diag(c1),'k-',label='${\\rm Analytical}$');
ax.plot(ls,np.diag(c2),'r-',label='${\\rm Jackknife}$');
for i in range(1,2):
    ax.plot(0.5*(ls[i:]+ls[:-i]),np.fabs(np.diag(c1,k=i)),'k--');
    ax.plot(0.5*(ls[i:]+ls[:-i]),np.fabs(np.diag(c2,k=i)),'r--');
ax.plot([-1,-1],[-1,-1],'k-',label='$i=j$')
ax.plot([-1,-1],[-1,-1],'k--',label='$i=j+1$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='lower left',ncol=2,fontsize=15,frameon=False)
ax.set_xlabel('$(\\ell_i+\\ell_j)/2$',fontsize=15)
ax.set_ylabel('$|{\\rm Cov}(\\ell_i,\\ell_j)|$',fontsize=15)
ax.tick_params(labelsize="x-large")
plt.savefig('notes/paper/cov_diag_'+gsample+'.pdf',bbox_inches='tight')
plt.show()
