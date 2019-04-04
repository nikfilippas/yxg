import sys
import numpy as np
from analysis.params import ParamRun
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.data import DataManager
from model.theory import get_theory

try:
    fname_params = sys.argv[1]
except IndexError:
    raise ValueError("Must provide param file name as command-line argument")

p = ParamRun(fname_params)

# Cosmology (Planck 2018)
cosmo = p.get_cosmo()

def print_covs(dcovs):
    import matplotlib.pyplot as plt
    for k in sorted(dcovs.keys()):
        c=dcovs[k].covar
        plt.figure()
        plt.title(k)
        plt.imshow(c/np.sqrt(np.diag(c)[:,None]*np.diag(c)[None,:]),
                   interpolation='nearest',vmin=-0.2,vmax=1)

    plt.figure()
    for k in sorted(dcovs.keys()):
        c=dcovs[k].covar
        plt.plot(np.diag(c),label=k)
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.show()

for v in p.get('data_vectors'):
    for n in ['data','model','jk','data_4pt','model_4pt','comb_m','comb_j']:
        v['covar_type']=n
        d = DataManager(p, v, cosmo)
    
        def th(pars):
            return get_theory(p, d, cosmo, **pars)

        lik = Likelihood(p.get('params'), d.data_vector, d.covar, th,
                         debug=p.get('mcmc')['debug'])
        sam = Sampler(lik.lnprob, lik.p0, p.get_sampler_prefix())
        sam.get_best_fit(update_p0=True)
        #    cov = sam.get_covariance()
        print(v['name'],n,sam.p0,lik.chi2(sam.p0),len(d.data_vector))
