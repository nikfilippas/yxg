import cosmotools as ct
import pyccl as ccl
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.81)

m200_arr=np.logspace(6,17,256)
z_arr=np.array([0.,0.1,0.3,0.5,1.0,1.5,2.0])
a_arr=1./(1+z_arr)

m500_arr=np.zeros([len(a_arr),len(m200_arr)])
c200_arr=np.zeros([len(a_arr),len(m200_arr)])
c500_arr=np.zeros([len(a_arr),len(m200_arr)])

def func_tobrent(c,d) :
    return (np.log(1+c)-c/(1+c))/(d*c*c*c)

print("Translating c200 into c500")
plt.figure()
for ia,a in enumerate(a_arr) :
    ha2=(cosmo['h']*ccl.h_over_h0(cosmo,a))**2
    for im,m200 in enumerate(m200_arr) :
        #At each redshift and at each value of m200
        r200=ct.R_Delta(cosmo,m200,a,Delta=200) #Compute r200
        c200=ct.concentration_duffy(m200,a) #Compute c200
        c200_arr[ia,im]=c200

        #Get c500 from c200 by solving:
        # [ ln(1+c500)-c500/(1+c500) ]/(500*c500**3) = [ ln(1+c200)-c200/(1+c200) ]/(200*c200**3)
        func0=func_tobrent(c200,200)
        def fzero(c) :
            return func_tobrent(c,500)-func0
        c500=brentq(fzero,c200/10.,c200*10,maxiter=1000)
        c500_arr[ia,im]=c500

        #That should be it, but we also want to see the translation between m200 and m500
        #for our own education
        r500=c500*r200/c200 #Compute r500
        m500=500*1.16217766E12*ha2*r500**3 #Compute m500
        m500_arr[ia,im]=m500

    plt.plot(m500_arr[ia],c500_arr[ia],c=cm.spring(ia/(len(a_arr)-1.))) #Plot both c-M relations
    plt.plot(m200_arr,c200_arr[ia],c=cm.winter(ia/(len(a_arr)-1.)))
plt.loglog();
plt.xlabel("$M/M_\\odot$",fontsize=15)
plt.ylabel("$c(M,a)$",fontsize=15)
plt.show()

#Now we want to put the new c_M relation in the same form as Duffy:
#   log(c(M,a)) = log(A) + B*log(M/M_pivot) - C*log(a)
#This is a simple linear least-squares problem that can be solved analytically.
kmat=np.array([np.ones_like(m500_arr).flatten(),
               np.log(m500_arr.flatten()/2.78164E12),
               -np.log((a_arr[:,None]*np.ones(len(m200_arr))[None,:]).flatten())])
bf=np.linalg.solve(np.dot(kmat,kmat.T),np.dot(kmat,np.log(c500_arr.flatten())))
print("New Duffy parameters (A,B,C) = ",(np.exp(bf[0]),bf[1],bf[2]))

#Check whether the new parameters reproduce the data sufficiently well
plt.figure()
c500b_arr=np.zeros([len(a_arr),len(m200_arr)])
for ia,a in enumerate(a_arr) :
    for im,m in enumerate(m200_arr) :
        m500=m500_arr[ia,im]
        c500=ct.concentration_duffy(m500,a,is_D500=True)
        c500b_arr[ia,im]=c500
    plt.plot(m500_arr[ia],c500_arr[ia],c=cm.spring(ia/(len(a_arr)-1.)))
    plt.plot(m500_arr[ia],c500b_arr[ia],'k--')
plt.loglog();
plt.xlabel("$M/M_\\odot$",fontsize=15)
plt.ylabel("$c(M,a)$",fontsize=15)
plt.show()
