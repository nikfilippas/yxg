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

for v in p.get('data_vectors'):
    d = DataManager(p, v, cosmo)

    def th(pars):
        return get_theory(p, d, cosmo, **pars)

    lik = Likelihood(p.get('params'), d.data_vector, d.covar, th,
                     debug=p.get('mcmc')['debug'])
    sam = Sampler(lik.lnprob, lik.p0, p.get_sampler_prefix())

    sam.get_best_fit(update_p0=True)
    cov = sam.get_covariance()
    print(sam.p0,
          np.sqrt(np.diag(cov)),
          lik.chi2(sam.p0),
          len(d.data_vector))
