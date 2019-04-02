import sys
import numpy as np
from analysis.params import ParamRun
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.data import DataManager
from model.theory import get_theory
import matplotlib.pyplot as plt
from model.power_spectrum import HalomodCorrection, hm_bias

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

zmeans = []
szmeans = []
bmeans = []
sbmeans = []
for v in p.get('data_vectors'):
    print(v['name'])

    # Construct data vector and covariance
    d = DataManager(p, v, cosmo)
    z, nz = np.loadtxt(d.tracers[0][0].dndz, unpack=True)
    zmean = np.sum(nz * z) / np.sum(nz)
    sigz = np.sqrt(np.sum(nz * (z - zmean)**2) / np.sum(nz))
    zmeans.append(zmean)
    szmeans.append(sigz)

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

    # Compute galaxy bias
    zarr = np.linspace(zmean - sigz, zmean + sigz, 10)
    bg = np.mean(hm_bias(cosmo, 1./(1 + zarr),
                         d.tracers[0][0].profile,
                         **(lik.build_kwargs(sam.p0))))
    by = np.mean(hm_bias(cosmo, 1./(1 + zarr),
                         d.tracers[1][1].profile,
                         **(lik.build_kwargs(sam.p0))))

    # Plot power spectra
    figs_cl = lik.plot_data(sam.p0, d, save_figures=True,
                            prefix=p.get_sampler_prefix(v['name']),
                            get_theory_1h=th1h, get_theory_2h=th2h)

    # Plot likelihood
    figs_ch = lik.plot_chain(sam.chain, save_figure=True,
                             prefix=p.get_sampler_prefix(v['name']))

    print(" Best-fit parameters:")
    for nn, vv, ss in zip(sam.parnames, sam.p0, np.std(sam.chain, axis=0)):
        print("  " + nn + " : %.3lE +- %.3lE" % (vv, ss))
        if nn == 'b_hydro':
            bmeans.append(vv)
            sbmeans.append(ss)
        chain = sam.chain
    print(" chi^2 = %lf" % (lik.chi2(sam.p0)))
    print(" n_data = %d" % (len(d.data_vector)))
    print(" b_g = %lf" % bg)
    print(" b_y = %.2lE" % by)

plt.figure()
plt.errorbar(zmeans, 1-np.array(bmeans),
             xerr=szmeans, yerr=sbmeans, fmt='ro')
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$1-b$', fontsize=15)
plt.savefig(p.get_sampler_prefix('b_hydro')+'all.pdf',
            bbox_inches='tight')
plt.show()
