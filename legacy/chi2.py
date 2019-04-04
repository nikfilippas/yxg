import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
from scipy.interpolate import interp1d
from analysis.beams import beam_y_planck,beam_hpix

#Cosmology (Planck 2018)
cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665)
kmax=1.
nside=512

def get_chi2(name,nameb,fname_dndz):
    fname_data_gg="output_test/cls_"+name+"_"+name+".npz"
    fname_data_gy="output_test/cls_y_milca_"+name+".npz"
    fname_cov_data_gggg="output_test/cov_data_"+name+"_"+name+"_"+name+"_"+name+".npz"
    fname_cov_data_gggy="output_test/cov_data_"+name+"_"+name+"_"+name+"_y_milca.npz"
    fname_cov_data_gygy="output_test/cov_data_"+name+"_y_milca_"+name+"_y_milca.npz"
    fname_cov_model_gggg="output_test/cov_model_"+name+"_"+name+"_"+name+"_"+name+".npz"
    fname_cov_model_gggy="output_test/cov_model_"+name+"_"+name+"_"+name+"_y_milca.npz"
    fname_cov_model_gygy="output_test/cov_model_"+name+"_y_milca_"+name+"_y_milca.npz"
    fname_cov_jk_gggg="output_test/cov_jk_"+name+"_"+name+"_"+name+"_"+name+".npz"
    fname_cov_jk_gggy="output_test/cov_jk_"+name+"_"+name+"_"+name+"_y_milca.npz"
    fname_cov_jk_gygy="output_test/cov_jk_"+name+"_y_milca_"+name+"_y_milca.npz"
    fname_cov_cbm_gggg="output_test/cov_comb_m_"+name+"_"+name+"_"+name+"_"+name+".npz"
    fname_cov_cbm_gggy="output_test/cov_comb_m_"+name+"_"+name+"_"+name+"_y_milca.npz"
    fname_cov_cbm_gygy="output_test/cov_comb_m_"+name+"_y_milca_"+name+"_y_milca.npz"
    fname_cov_cbj_gggg="output_test/cov_comb_j_"+name+"_"+name+"_"+name+"_"+name+".npz"
    fname_cov_cbj_gggy="output_test/cov_comb_j_"+name+"_"+name+"_"+name+"_y_milca.npz"
    fname_cov_cbj_gygy="output_test/cov_comb_j_"+name+"_y_milca_"+name+"_y_milca.npz"
    fname_dcov_gggg="output_test/dcov_1h4pt_"+name+"_"+name+"_"+name+"_"+name+".npz"
    fname_dcov_gggy="output_test/dcov_1h4pt_"+name+"_"+name+"_"+name+"_y_milca.npz"
    fname_dcov_gygy="output_test/dcov_1h4pt_"+name+"_y_milca_"+name+"_y_milca.npz"
    fname_theory="output_test/cl_th_"+name+".npz"

    cv_d_gggg=np.load(fname_cov_data_gggg)['cov']
    cv_d_gggy=np.load(fname_cov_data_gggy)['cov']
    cv_d_gygy=np.load(fname_cov_data_gygy)['cov']
    cv_m_gggg=np.load(fname_cov_model_gggg)['cov']
    cv_m_gggy=np.load(fname_cov_model_gggy)['cov']
    cv_m_gygy=np.load(fname_cov_model_gygy)['cov']
    cv_j_gggg=np.load(fname_cov_jk_gggg)['cov']
    cv_j_gggy=np.load(fname_cov_jk_gggy)['cov']
    cv_j_gygy=np.load(fname_cov_jk_gygy)['cov']
    cv_cm_gggg=np.load(fname_cov_cbm_gggg)['cov']
    cv_cm_gggy=np.load(fname_cov_cbm_gggy)['cov']
    cv_cm_gygy=np.load(fname_cov_cbm_gygy)['cov']
    cv_cj_gggg=np.load(fname_cov_cbj_gggg)['cov']
    cv_cj_gggy=np.load(fname_cov_cbj_gggy)['cov']
    cv_cj_gygy=np.load(fname_cov_cbj_gygy)['cov']
    dcv_gggg=np.load(fname_dcov_gggg)['cov']
    dcv_gggy=np.load(fname_dcov_gggy)['cov']
    dcv_gygy=np.load(fname_dcov_gygy)['cov']
    
    z,nz=np.loadtxt(fname_dndz,unpack=True); zmean=np.sum(z*nz)/np.sum(nz);
    lmax=kmax*ccl.comoving_radial_distance(cosmo,1/(1+zmean))-0.5
    
    d=np.load(fname_data_gg);
    cl_d_gg=d['cls']-d['nls']
    ls_gg=d['ls']
    wgg=d['windows']
    m_gg=ls_gg<lmax

    d=np.load(fname_data_gy);
    cl_d_gy=d['cls']-d['nls']
    ls_gy=d['ls']
    wgy=d['windows']
    m_gy=ls_gy<lmax

    d=np.load(fname_theory)
    ls=np.arange(3*nside)
    cl_t_gg=np.dot(wgg,d['clgg'])
    cl_t_gy=np.dot(wgy,d['clgy'])
    plt.figure()
    plt.errorbar(ls_gg[m_gg],cl_d_gg[m_gg],yerr=np.sqrt(np.diag(cv_d_gggg+dcv_gggg))[m_gg],fmt='r.')
    plt.plot(ls_gg[m_gg],cl_t_gg[m_gg],'r-')
    plt.loglog()
    plt.figure()
    plt.errorbar(ls_gy[m_gy],cl_d_gy[m_gy],yerr=np.sqrt(np.diag(cv_d_gygy+dcv_gygy))[m_gy],fmt='r.')
    plt.plot(ls_gy[m_gy],cl_t_gy[m_gy],'r-')
    plt.loglog()

    cl_d_tot=np.concatenate((cl_d_gg,cl_d_gy))
    cl_t_tot=np.concatenate((cl_t_gg,cl_t_gy))
    m_tot=np.concatenate((m_gg,m_gy))
    ngg=len(ls_gg); ngy=len(ls_gy)
    cv_d_tot=np.zeros([ngg+ngy,ngg+ngy])
    cv_d_tot[:ngg,:][:,:ngg]=cv_d_gggg
    cv_d_tot[:ngg,:][:,ngg:]=cv_d_gggy
    cv_d_tot[ngg:,:][:,:ngg]=cv_d_gggy.T
    cv_d_tot[ngg:,:][:,ngg:]=cv_d_gygy
    cv_m_tot=np.zeros([ngg+ngy,ngg+ngy])
    cv_m_tot[:ngg,:][:,:ngg]=cv_m_gggg
    cv_m_tot[:ngg,:][:,ngg:]=cv_m_gggy
    cv_m_tot[ngg:,:][:,:ngg]=cv_m_gggy.T
    cv_m_tot[ngg:,:][:,ngg:]=cv_m_gygy
    cv_j_tot=np.zeros([ngg+ngy,ngg+ngy])
    cv_j_tot[:ngg,:][:,:ngg]=cv_j_gggg
    cv_j_tot[:ngg,:][:,ngg:]=cv_j_gggy
    cv_j_tot[ngg:,:][:,:ngg]=cv_j_gggy.T
    cv_j_tot[ngg:,:][:,ngg:]=cv_j_gygy
    cv_cm_tot=np.zeros([ngg+ngy,ngg+ngy])
    cv_cm_tot[:ngg,:][:,:ngg]=cv_cm_gggg
    cv_cm_tot[:ngg,:][:,ngg:]=cv_cm_gggy
    cv_cm_tot[ngg:,:][:,:ngg]=cv_cm_gggy.T
    cv_cm_tot[ngg:,:][:,ngg:]=cv_cm_gygy
    cv_cj_tot=np.zeros([ngg+ngy,ngg+ngy])
    cv_cj_tot[:ngg,:][:,:ngg]=cv_cj_gggg
    cv_cj_tot[:ngg,:][:,ngg:]=cv_cj_gggy
    cv_cj_tot[ngg:,:][:,:ngg]=cv_cj_gggy.T
    cv_cj_tot[ngg:,:][:,ngg:]=cv_cj_gygy
    dcv_tot=np.zeros([ngg+ngy,ngg+ngy])
    dcv_tot[:ngg,:][:,:ngg]=dcv_gggg
    dcv_tot[:ngg,:][:,ngg:]=dcv_gggy
    dcv_tot[ngg:,:][:,:ngg]=dcv_gggy.T
    dcv_tot[ngg:,:][:,ngg:]=dcv_gygy
    
    def chi2(dx,cv,njk=None):
        icv=np.linalg.inv(cv)
        if njk is not None:
            hartfac=(njk-len(dx)-2.)/(njk-1.)
            icv*=hartfac
        return np.einsum('i,ij,j',dx,icv,dx)
    print(name+", gg , cbm  , %lf %d"%(chi2((cl_d_gg-cl_t_gg)[m_gg],cv_cm_gggg[m_gg,:][:,m_gg]),np.sum(m_gg)))
    print(name+", gg , cbj  , %lf %d"%(chi2((cl_d_gg-cl_t_gg)[m_gg],cv_cj_gggg[m_gg,:][:,m_gg],njk=461),np.sum(m_gg)))
    print(name+", gg , jk   , %lf %d"%(chi2((cl_d_gg-cl_t_gg)[m_gg],cv_j_gggg[m_gg,:][:,m_gg],njk=461),np.sum(m_gg)))
    print(name+", gg , data , %lf %d"%(chi2((cl_d_gg-cl_t_gg)[m_gg],cv_d_gggg[m_gg,:][:,m_gg]),np.sum(m_gg)))
    print(name+", gg , model, %lf %d"%(chi2((cl_d_gg-cl_t_gg)[m_gg],cv_m_gggg[m_gg,:][:,m_gg]),np.sum(m_gg)))
    print(name+", gg , d+1h , %lf %d"%(chi2((cl_d_gg-cl_t_gg)[m_gg],(cv_d_gggg+dcv_gggg)[m_gg,:][:,m_gg]),np.sum(m_gg)))
    print(name+", gg , m+1h , %lf %d"%(chi2((cl_d_gg-cl_t_gg)[m_gg],(cv_m_gggg+dcv_gggg)[m_gg,:][:,m_gg]),np.sum(m_gg)))
    print(name+", gy , cbm   , %lf %d"%(chi2((cl_d_gy-cl_t_gy)[m_gy],cv_cm_gygy[m_gy,:][:,m_gy]),np.sum(m_gy)))
    print(name+", gy , cbj   , %lf %d"%(chi2((cl_d_gy-cl_t_gy)[m_gy],cv_cj_gygy[m_gy,:][:,m_gy],njk=461),np.sum(m_gy)))
    print(name+", gy , jk   , %lf %d"%(chi2((cl_d_gy-cl_t_gy)[m_gy],cv_j_gygy[m_gy,:][:,m_gy],njk=461),np.sum(m_gy)))
    print(name+", gy , data , %lf %d"%(chi2((cl_d_gy-cl_t_gy)[m_gy],cv_d_gygy[m_gy,:][:,m_gy]),np.sum(m_gy)))
    print(name+", gy , model, %lf %d"%(chi2((cl_d_gy-cl_t_gy)[m_gy],cv_m_gygy[m_gy,:][:,m_gy]),np.sum(m_gy)))
    print(name+", gy , d+1h , %lf %d"%(chi2((cl_d_gy-cl_t_gy)[m_gy],(cv_d_gygy+dcv_gygy)[m_gy,:][:,m_gy]),np.sum(m_gy)))
    print(name+", gy , m+1h , %lf %d"%(chi2((cl_d_gy-cl_t_gy)[m_gy],(cv_m_gygy+dcv_gygy)[m_gy,:][:,m_gy]),np.sum(m_gy)))
    print(name+", tot, cbm  , %lf %d"%(chi2((cl_d_tot-cl_t_tot)[m_tot],cv_cm_tot[m_tot,:][:,m_tot]),np.sum(m_tot)))
    print(name+", tot, cbj  , %lf %d"%(chi2((cl_d_tot-cl_t_tot)[m_tot],cv_cj_tot[m_tot,:][:,m_tot],njk=461),np.sum(m_tot)))
    print(name+", tot, jk   , %lf %d"%(chi2((cl_d_tot-cl_t_tot)[m_tot],cv_j_tot[m_tot,:][:,m_tot],njk=461),np.sum(m_tot)))
    print(name+", tot, data , %lf %d"%(chi2((cl_d_tot-cl_t_tot)[m_tot],cv_d_tot[m_tot,:][:,m_tot]),np.sum(m_tot)))
    print(name+", tot, model, %lf %d"%(chi2((cl_d_tot-cl_t_tot)[m_tot],cv_m_tot[m_tot,:][:,m_tot]),np.sum(m_tot)))
    print(name+", tot, d+1h , %lf %d"%(chi2((cl_d_tot-cl_t_tot)[m_tot],(cv_d_tot+dcv_tot)[m_tot,:][:,m_tot]),np.sum(m_tot)))
    print(name+", tot, m+1h , %lf %d"%(chi2((cl_d_tot-cl_t_tot)[m_tot],(cv_m_tot+dcv_tot)[m_tot,:][:,m_tot]),np.sum(m_tot)))
    print(" ")

    plt.figure()
    plt.plot(ls_gg[m_gg],np.sqrt(np.diag(cv_cm_gggg))[m_gg],'k-')
    plt.plot(ls_gg[m_gg],np.sqrt(np.diag(cv_cj_gggg))[m_gg],'k--')
    plt.plot(ls_gg[m_gg],np.sqrt(np.diag(cv_j_gggg))[m_gg],'k-.')
    plt.plot(ls_gg[m_gg],np.sqrt(np.diag(cv_d_gggg))[m_gg],'r-')
    plt.plot(ls_gg[m_gg],np.sqrt(np.diag(cv_m_gggg))[m_gg],'r--')
    plt.plot(ls_gg[m_gg],np.sqrt(np.diag(cv_m_gggg+dcv_gggg))[m_gg],'r-.')
    plt.plot(ls_gy[m_gy],np.sqrt(np.diag(cv_cm_gygy))[m_gy],'c-')
    plt.plot(ls_gy[m_gy],np.sqrt(np.diag(cv_cj_gygy))[m_gy],'c--')
    plt.plot(ls_gy[m_gy],np.sqrt(np.diag(cv_j_gygy))[m_gy],'c-.')
    plt.plot(ls_gy[m_gy],np.sqrt(np.diag(cv_d_gygy))[m_gy],'b-')
    plt.plot(ls_gy[m_gy],np.sqrt(np.diag(cv_m_gygy))[m_gy],'b--')
    plt.plot(ls_gy[m_gy],np.sqrt(np.diag(cv_m_gygy+dcv_gygy))[m_gy],'b-.')
    plt.loglog();
    plt.show()


get_chi2("2mpz","2mpz","data/dndz/2MPZ_bin1.txt")
for i in range(5):
    get_chi2("wisc%d"%(i+1),"wisc_b%d"%(i+1),"data/dndz/WISC_bin%d.txt"%(i+1))
