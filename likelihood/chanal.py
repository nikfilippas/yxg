import os
os.chdir("../")
import numpy as np
from analysis.params import ParamRun
from model.data import DataManager
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.theory import get_theory
from model.power_spectrum import HalomodCorrection
from model.utils import selection_planck_erf, selection_planck_tophat


def chan(fname_params):
    """
    Given a parameter file, looks up corresponding sampler and calculates
    best-fit parameters. Outputs a dictionary of best-fit parameters,
    chi-squared and dof.
    """
    def th(pars):
        return get_theory(p, d, cosmo, hm_correction=hm_correction,
                          selection=sel, **pars)

    def halmodcor(p, cosmo):
        # Include halo model correction if needed
        if p.get('mcmc').get('hm_correct'):
            hm_correction = HalomodCorrection(cosmo)
        else:
            hm_correction = None
        return hm_correction

    def selfunc(p):
        # Include selection function if needed
        sel = p.get('mcmc').get('selection_function')
        if sel is not None:
            if sel == 'erf':
                sel = selection_planck_erf
            elif sel == 'tophat':
                sel = selection_planck_tophat
            elif sel == 'none':
                sel = None
        return sel

    p = ParamRun(fname_params)
    cosmo = p.get_cosmo()
    sel = selfunc(p)
    hm_correction = halmodcor(p, cosmo)

    params, chi2, dof = [[] for i in range(3)]
    for s, v in enumerate(p.get("data_vectors")):

        # Construct data vector and covariance
        d = DataManager(p, v, cosmo, all_data=False)
        z, nz = np.loadtxt(d.tracers[0][0].dndz, unpack=True)

        lik = Likelihood(p.get('params'), d.data_vector, d.covar, th)
        sam = Sampler(lik.lnprob, lik.p0, lik.p_free_names,
                      p.get_sampler_prefix(v['name']), p.get('mcmc'))

        # Read chains and best-fit
        sam.get_chain()
        sam.update_p0(sam.chain[np.argmax(sam.probs)])
        params.append(lik.build_kwargs(sam.p0))
        chi2.append(lik.chi2(sam.p0))
        dof.append(len(lik.dv))

    return params, (chi2, dof)