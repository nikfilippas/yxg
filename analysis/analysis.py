from __future__ import print_function
from optparse import OptionParser
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
import os
from scipy.interpolate import interp1d

def opt_callback(option, opt, value, parser):
  setattr(parser.values, option.dest, value.split(','))

parser=OptionParser()
parser.add_option("--nside",dest='nside',default=512,type=int,help="HEALPix Nside resolution parameter")
parser.add_option("--recreate",dest='recreate',default=False,action='store_true',
                  help="Recompute power spectra and covariances?")
parser.add_option("--linlog_binning",dest='use_linlog',default=False,action='store_true',
                  help="Use hybrid lin-log binning for bandpowers")
parser.add_option("--nlb",dest='nlb',default=20,type=int,
                  help="If using linear bandpowers, this is the bandpowers width")
parser.add_option("--sz-mask",dest='sz_mask',default="data/maps/mask_planck60.fits",type=str,
                  help="Path to file containing the mask that should be used for the SZ maps.")
(o, args) = parser.parse_args()

#TODO: y beam
nside_use=o.nside
recreate=o.recreate
use_linlog=o.use_linlog
nlb_use=o.nlb
fname_sz_mask=o.sz_mask

predir_out='out'
predir_out+="_ns%d"%nside_use
if use_linlog :
    predir_out+='_linlog'
else :
    predir_out+='_nb%d'%nlb_use
os.system("mkdir -p "+predir_out+'/plots')
    
def read_map(name,fname,nside_out,field=0) :
    print("  Reading "+name)
    return hp.ud_grade(hp.read_map(fname,verbose=False),nside_out=nside_out)

def get_field(name,fname,nside,mask,is_ndens=False) :
    mask_bn=np.ones_like(mask); mask_bn[mask<=0]=0; #Get binary mask
    mp=read_map(name,fname,nside) # Read map
    mp*=mask_bn #Remove masked pixels
    ndens=0
    if is_ndens : #Get overdensity if needed
        print("Normalizing")
        mp*=mask;
        mean_g=np.sum(mp)/np.sum(mask)
        ndens=mean_g*12*nside**2/(4*np.pi)
        mp=mask*(mp/mean_g-1.)

    return nmt.NmtField(mask,[mp]),ndens

def get_cl(k1,k2,nside,do_recreate=False) :
    fname_out=predir_out+"/cl_"+k1+"_"+k2
    if (os.path.isfile(fname_out+'.npz')) and (not do_recreate) :
        print("Reading Cl "+k1+" "+k2)
        d=np.load(fname_out+'.npz')
        return d['leff'],d['cell'],d['nell']
    if k1 not in fields.keys() :
        fields[k1],ndenss[k1]=get_field(k1,fn_maps[k1],nside,masks[k1],is_ndens=isdens[k1])
    if k2 not in fields.keys() :
        fields[k2],ndenss[k2]=get_field(k2,fn_maps[k2],nside,masks[k2],is_ndens=isdens[k2])
    f1=fields[k1]; f2=fields[k2];

    print("  "+k1+" "+k2)
    w=nmt.NmtWorkspace()
    fname_workspace=predir_out+'/w_'+msktyp[k1]+'_'+msktyp[k2]+'.dat'
    if os.path.isfile(fname_workspace) :
        w.read_from(fname_workspace)
    else :
        w.compute_coupling_matrix(f1,f2,bb)
        w.write_to(fname_workspace)
    cl=w.decouple_cell(nmt.compute_coupled_cell(f1,f2))[0]
    larr=bb.get_effective_ells()

    if (k1==k2) and (isdens[k1]) :
        nl=w.decouple_cell(w.couple_cell([np.ones(3*nside)/ndenss[k1]]))[0]
        print(k1,np.mean(nl),1/ndenss[k1])
    else :
        nl=np.zeros_like(cl)
    np.savez(fname_out,leff=larr,cell=cl,nell=nl)
    return larr,cl,nl

def get_covariance(ka1,ka2,kb1,kb2,ca1b1,ca1b2,ca2b1,ca2b2,do_recreate=False) :
    fname_out=predir_out+'/cov_'+ka1+'_'+ka2+'_'+kb1+'_'+kb2
    if (os.path.isfile(fname_out+'.npz')) and (not do_recreate) :
        d=np.load(fname_out+'.npz')
        return d['cov']
    print(ka1,ka2,kb1,kb2)
    fname_cworkspace=predir_out+'/cw_'+msktyp[ka1]+'_'+msktyp[ka2]+'_'+msktyp[kb1]+'_'+msktyp[kb2]
    fname_cworkspace+='.dat'
    cw=nmt.NmtCovarianceWorkspace()
    if os.path.isfile(fname_cworkspace) :
        cw.read_from(fname_cworkspace)
    else :
        wa=nmt.NmtWorkspace();
        wa.read_from(predir_out+'/w_'+msktyp[ka1]+'_'+msktyp[ka2]+'.dat')
        wb=nmt.NmtWorkspace();
        wb.read_from(predir_out+'/w_'+msktyp[kb1]+'_'+msktyp[kb2]+'.dat')
        cw.compute_coupling_coefficients(wa,wb)
        cw.write_to(fname_cworkspace)
    cov_here=nmt.gaussian_covariance(cw,ca1b1,ca1b2,ca2b1,ca2b2)

    np.savez(fname_out,cov=cov_here)
    return cov_here

    
map_names=['2mpz','wisc_b1','wisc_b2','wisc_b3','wisc_b4','wisc_b5',
           'sdss_b1','sdss_b2','sdss_b3','sdss_b4','sdss_b5',
           'sdss_b6','sdss_b7','sdss_b8','sdss_b9','y_milca','y_nilc']
map_types={'2mpz':'g','wisc_b1':'g','wisc_b2':'g',
           'wisc_b3':'g','wisc_b4':'g','wisc_b5':'g',
           'sdss_b1':'g','sdss_b2':'g','sdss_b3':'g',
           'sdss_b4':'g','sdss_b5':'g','sdss_b6':'g',
           'sdss_b7':'g','sdss_b8':'g','sdss_b9':'g',
           'y_milca':'y','y_nilc':'y'}
nmaps=len(map_names)
fn_maps={'2mpz':'data/maps/2mpz_05_01_512.fits',
         'wisc_b1':'data/maps/2dstarsub_WISC_cleaned_public.bin_0.1_z_0.15.Pix512.fits',
         'wisc_b2':'data/maps/2dstarsub_WISC_cleaned_public.bin_0.15_z_0.2.Pix512.fits',
         'wisc_b3':'data/maps/2dstarsub_WISC_cleaned_public.bin_0.2_z_0.25.Pix512.fits',
         'wisc_b4':'data/maps/2dstarsub_WISC_cleaned_public.bin_0.25_z_0.3.Pix512.fits',
         'wisc_b5':'data/maps/2dstarsub_WISC_cleaned_public.bin_0.3_z_0.35.Pix512.fits',
         'sdss_b1':'data/maps/SDSS.DR12.0_1photoz0_15_rLess21_512.fits',
         'sdss_b2':'data/maps/SDSS.DR12.0_15photoz0_2_rLess21_512.fits',
         'sdss_b3':'data/maps/SDSS.DR12.0_2photoz0_25_rLess21_512.fits',
         'sdss_b4':'data/maps/SDSS.DR12.0_25photoz0_3_rLess21_512.fits',
         'sdss_b5':'data/maps/SDSS.DR12.0_3photoz0_35_rLess21_512.fits',
         'sdss_b6':'data/maps/SDSS.DR12.0_35photoz0_4_rLess21_512.fits',
         'sdss_b7':'data/maps/SDSS.DR12.0_4photoz0_5_rLess21_512.fits',
         'sdss_b8':'data/maps/SDSS.DR12.0_5photoz0_6_rLess21_512.fits',
         'sdss_b9':'data/maps/SDSS.DR12.0_6photoz0_7_rLess21_512.fits',
         'y_milca':'data/maps/milca_ymaps.fits',
         'y_nilc':'data/maps/nilc_ymaps.fits'}

print("Bandpowers")
if use_linlog :
    #Linlog binning
    lsplit=52
    l_edges_lin_custom=np.linspace(2,lsplit,(lsplit-2)//10+1).astype(int)
    l_edges_log_custom=np.unique(np.logspace(np.log10(lsplit),np.log10(3*nside_use-1),20).astype(int))
    l_edges_custom=np.concatenate((l_edges_lin_custom,l_edges_log_custom[1:]))
    larr=np.arange(3*nside_use)
    weights=np.ones(len(larr))
    bpws=-1+np.zeros(len(larr),dtype=int)
    for i in range(len(l_edges_custom)-1) :
        bpws[l_edges_custom[i]:l_edges_custom[i+1]]=i
    bb=nmt.NmtBin(nside_use,ells=larr,bpws=bpws,weights=weights)
else :
    #Linear binning
    bb=nmt.NmtBin(nside_use,nlb=nlb_use)
l_eff=bb.get_effective_ells()

print("Reading masks")
mask_lowz=read_map('mask_lowz','data/maps/mask_v3.fits',nside_use)
mask_sdss=read_map('mask_sdss','data/maps/BOSS_dr12_mask256_v2.fits',nside_use)
mask_y=read_map('mask_y',fname_sz_mask,nside_use)
masks={'2mpz':mask_lowz,'wisc_b1':mask_lowz,'wisc_b2':mask_lowz,
       'wisc_b3':mask_lowz,'wisc_b4':mask_lowz,'wisc_b5':mask_lowz,
       'sdss_b1':mask_sdss,'sdss_b2':mask_sdss,'sdss_b3':mask_sdss,
       'sdss_b4':mask_sdss,'sdss_b5':mask_sdss,'sdss_b6':mask_sdss,
       'sdss_b7':mask_sdss,'sdss_b8':mask_sdss,'sdss_b9':mask_sdss,
       'y_milca':mask_y,'y_nilc':mask_y}
isdens={'2mpz':True,'wisc_b1':True,'wisc_b2':True,
        'wisc_b3':True,'wisc_b4':True,'wisc_b5':True,
        'sdss_b1':True,'sdss_b2':True,'sdss_b3':True,
        'sdss_b4':True,'sdss_b5':True,'sdss_b6':True,
        'sdss_b7':True,'sdss_b8':True,'sdss_b9':True,
        'y_milca':False,'y_nilc':False}
msktyp={'2mpz':'lowz','wisc_b1':'lowz','wisc_b2':'lowz',
        'wisc_b3':'lowz','wisc_b4':'lowz','wisc_b5':'lowz',
        'sdss_b1':'sdss','sdss_b2':'sdss','sdss_b3':'sdss',
        'sdss_b4':'sdss','sdss_b5':'sdss','sdss_b6':'sdss',
        'sdss_b7':'sdss','sdss_b8':'sdss','sdss_b9':'sdss',
        'y_milca':'cmb','y_nilc':'cmb'}

fields={}
ndenss={}

print("Computing power spectra")
nell=len(l_eff)
cls_vec=[]
nls_vec=[]
cls_th=np.zeros([nmaps,nmaps,3*nside_use])
lth=np.arange(3*nside_use)
for i1 in range(len(map_names)) :
    for i2 in range(i1,len(map_names)) :
        _,cl,nl=get_cl(map_names[i1],map_names[i2],nside_use,do_recreate=recreate)
        clf=interp1d(l_eff,cl,bounds_error=False,fill_value=0)
        clth=clf(lth); clth[lth<=l_eff[0]]=cl[0]; clth[lth>=l_eff[-1]]=cl[-1]
        cls_th[i1,i2,:]=clth
        cls_vec.append(cl)
        nls_vec.append(nl)
        if i1!=i2 :
            cls_th[i2,i1,:]=clth
cls_vec=np.array(cls_vec)
nls_vec=np.array(nls_vec)
ncross=(nmaps*(nmaps+1))//2

print("Computing covariance matrix")
covar=np.zeros([ncross*nell,ncross*nell])
ix=0
for i1 in range(len(map_names)) :
    for i2 in range(i1,len(map_names)) :
        jx=0
        print(ix)
        for j1 in range(len(map_names)) :
            for j2 in range(j1,len(map_names)) :
                cov=get_covariance(map_names[i1],map_names[i2],
                                   map_names[j1],map_names[j2],
                                   cls_th[i1,j1],cls_th[i1,j2],
                                   cls_th[i2,j1],cls_th[i2,j2],do_recreate=recreate)
                covar[ix*nell:(ix+1)*nell,:][:,jx*nell:(jx+1)*nell]=cov
                jx+=1
        ix+=1
covar=(0.5*(covar+covar.T)).reshape([ncross,nell,ncross,nell])

print("Plotting")
import matplotlib
cm_wisc=matplotlib.cm.get_cmap('Reds')
cm_sdss=matplotlib.cm.get_cmap('Blues')
cols=[]
labels=[]
cols.append('k')
labels.append('2MPZ')
for i in np.arange(5) :
    cols.append(cm_wisc(0.2+((i+1.)/5.)*0.8))
    labels.append('WISC Z%d'%(i+1))
for i in np.arange(9) :
    cols.append(cm_sdss(0.2+((i+1.)/9.)*0.8))
    labels.append('SDSS Z%d'%(i+1))


#AUTO
mask_cls=np.zeros(ncross,dtype=bool)
ix=0
for i1 in range(len(map_names)) :
    for i2 in range(i1,len(map_names)) :
        if (map_types[map_names[i1]]=='g') and (map_names[i2]==map_names[i1]) :
            mask_cls[ix]=True
        ix+=1

cls_vec_masked=cls_vec[mask_cls,:]
nls_vec_masked=nls_vec[mask_cls,:]
covar_masked=covar[mask_cls,:,:,:][:,:,mask_cls,:]

plt.figure()
for i in np.arange(np.sum(mask_cls)) :
    print(i,len(ndenss))
    cl=cls_vec_masked[i]
    nl=nls_vec_masked[i]
    ecl=np.sqrt(np.diag(covar_masked[i,:,i,:]))
    plt.errorbar(l_eff,cl-nl,yerr=ecl,fmt='-',c=cols[i],label=labels[i])
    plt.plot(l_eff,nl,ls='--',c=cols[i])#,label=labels[i])
plt.legend(loc='lower left',ncol=3);
plt.xscale('log'); plt.yscale('log'); #plt.ylim([5E-15,5E-10])
plt.xlabel('$\\ell$',fontsize=15); plt.ylabel('$C^{gg}_\\ell$',fontsize=15)
plt.savefig(predir_out+'/plots/clgg.pdf',bbox_inches='tight')

def get_corr(c) :
    return c#/np.sqrt(np.diag(c)[:,None]*np.diag(c)[None,:])

cov=covar_masked.reshape([15*nell,15*nell])
plt.figure(); plt.imshow(np.log10(np.fabs(get_corr(cov))),interpolation='nearest')
plt.savefig(predir_out+'/plots/covar_auto.pdf',bbox_inches='tight')

#MILCA
mask_cls=np.zeros(ncross,dtype=bool)
ix=0
for i1 in range(len(map_names)) :
    for i2 in range(i1,len(map_names)) :
        if (map_types[map_names[i1]]=='g') and (map_names[i2]=='y_milca') :
            mask_cls[ix]=True
        ix+=1

cls_vec_masked=cls_vec[mask_cls,:]
covar_masked=covar[mask_cls,:,:,:][:,:,mask_cls,:]
print(cls_vec_masked.shape,covar_masked.shape)

plt.figure()
for i in np.arange(np.sum(mask_cls)) :
    cl=cls_vec_masked[i]
    ecl=np.sqrt(np.diag(covar_masked[i,:,i,:]))
    plt.errorbar(l_eff,cl,yerr=ecl,fmt='-',c=cols[i],label=labels[i])
plt.legend(loc='lower left',ncol=3);
plt.xscale('log'); plt.yscale('log'); plt.ylim([5E-15,5E-10])
plt.xlabel('$\\ell$',fontsize=15); plt.ylabel('$C^{yg}_\\ell$',fontsize=15)
plt.savefig(predir_out+'/plots/clyg_milca.pdf',bbox_inches='tight')

cov=covar_masked.reshape([15*nell,15*nell])
plt.figure(); plt.imshow(np.log10(np.fabs(get_corr(cov))),interpolation='nearest')
plt.savefig(predir_out+'/plots/covar_milca.pdf',bbox_inches='tight')

#NILC
mask_cls=np.zeros(ncross,dtype=bool)
ix=0
for i1 in range(len(map_names)) :
    for i2 in range(i1,len(map_names)) :
        if (map_types[map_names[i1]]=='g') and (map_names[i2]=='y_nilc') :
            mask_cls[ix]=True
        ix+=1

cls_vec_masked=cls_vec[mask_cls,:]
covar_masked=covar[mask_cls,:,:,:][:,:,mask_cls,:]
print(cls_vec_masked.shape,covar_masked.shape)

plt.figure()
for i in np.arange(np.sum(mask_cls)) :
    cl=cls_vec_masked[i]
    ecl=np.sqrt(np.diag(covar_masked[i,:,i,:]))
    plt.errorbar(l_eff,cl,yerr=ecl,fmt='-',c=cols[i],label=labels[i])
plt.legend(loc='lower left',ncol=3);
plt.xscale('log'); plt.yscale('log'); plt.ylim([5E-15,5E-10])
plt.xlabel('$\\ell$',fontsize=15); plt.ylabel('$C^{yg}_\\ell$',fontsize=15)
plt.savefig(predir_out+'/plots/clyg_nilc.pdf',bbox_inches='tight')

cov=covar_masked.reshape([15*nell,15*nell])
plt.figure(); plt.imshow(np.log10(np.fabs(get_corr(cov))),interpolation='nearest')
plt.savefig(predir_out+'/plots/covar_nilc.pdf',bbox_inches='tight')
plt.show()
