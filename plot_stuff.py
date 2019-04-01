import sys
import numpy as np
from analysis.params import ParamRun
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.data import DataManager
from model.theory import get_theory
import matplotlib.pyplot as plt
from model.power_spectrum import HalomodCorrection

try:
    fname_params = sys.argv[1]
except IndexError:
    raise ValueError("Must provide param file name as command-line argument")

p = ParamRun(fname_params)

# Cosmology (Planck 2018)
cosmo = p.get_cosmo()

# Include halo model correction if needed
if p.get('mcmc').get('hm_correct'):
    hm_correction = HalomodCorrection(cosmo)
else:
    hm_correction = None

for v in p.get('data_vectors'):
    print(v['name'])

    # Construct data vector and covariance
    d = DataManager(p, v, cosmo)

    # Theory predictor wrapper
    def th(pars):
        return get_theory(p, d, cosmo, return_separated=False,
                          hm_correction=hm_correction,
                          **pars)

    def th1h(pars):
        return get_theory(p, d, cosmo, return_separated=False,
                          hm_correction=hm_correction,
                          include_2h=False, include_1h=True,
                          **pars)

    def th2h(pars):
        return get_theory(p, d, cosmo, return_separated=False,
                          hm_correction=hm_correction,
                          include_2h=True, include_1h=False,
                          **pars)

    # Set up likelihood
    lik = Likelihood(p.get('params'), d.data_vector, d.covar, th,
                     debug=p.get('mcmc')['debug'])

    # Set up sampler
    sam = Sampler(lik.lnprob, lik.p0, lik.p_free_names,
                  p.get_sampler_prefix(v['name']),
                  p.get('mcmc'))

    # Read chains and best-fit
    sam.get_chain()
    sam.update_p0(sam.chain[np.argmax(sam.probs)])

    # Plot power spectra
    figs_cl = lik.plot_data(sam.p0, d, save_figures=True,
                            prefix=p.get_sampler_prefix(v['name']),
                            get_theory_1h=th1h, get_theory_2h=th2h)

    # Plot likelihood
    figs_ch = lik.plot_chain(sam.chain, save_figure=True,
                             prefix=p.get_sampler_prefix(v['name']))

    print(" Best-fit parameters:")
    for nn,vv,ss in zip(sam.parnames, sam.p0, np.std(sam.chain,axis=0)):
        print("  " + nn + " : %.3lE +- %.3lE" %(vv,ss))
    print(" chi^2 = %lf" % (lik.chi2(sam.p0)))
    print(" n_data = %d" % (len(d.data_vector)))
plt.show()
