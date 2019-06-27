import sys
sys.path.append("../../")
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from analysis.params import ParamRun
from model.data import DataManager
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.theory import get_theory
from model.power_spectrum import HalomodCorrection, hm_bias
from model.utils import selection_planck_erf, selection_planck_tophat


def get_dndz(fname, width):
    """Get the modified galaxy number counts."""
    zd, Nd = np.loadtxt(fname, unpack=True)

    Nd /= simps(Nd, x=zd)
    zavg = np.average(zd, weights=Nd)
    nzf = interp1d(zd, Nd, kind="cubic", bounds_error=False, fill_value=0)

    Nd_new = nzf(zavg + (1/width)*(zd-zavg))
    return zd, Nd_new


def chan(fname_params, diff=False, probability=True):
    """
    Given a parameter file, looks up corresponding sampler and calculates
    best-fit parameters, and min & max values.
    Outputs a dictionary of best-fit parameters, chi-squared and dof.

    Args:
        fname_params (str): Name of parameter file.
        diff (bool): If True, return differences `vv-vmin`, `vmax-vv`.
                     If False, return min and max values.
        probability (bool): Cut using probability or stand-alone parameters.

    Returns:
        params (dict): Dictionary of values. "name" : [50, 16, 84] probability-
                       wise percentiles.
        chi2, dof (float): Chi-squared of best-fit and degrees of freedom.
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
        lik = Likelihood(p.get('params'), d.data_vector, d.covar, th)
        sam = Sampler(lik.lnprob, lik.p0, lik.p_free_names,
                      p.get_sampler_prefix(v['name']), p.get('mcmc'))

        # Read chains and best-fit
        sam.get_chain()

        if probability:
            sam.update_p0(sam.chain[np.argmax(sam.probs)])
            P_cut = np.percentile(sam.probs, 68)
            chain_cut = sam.chain[sam.probs > P_cut]
            vmin = np.min(chain_cut, axis=0)
            vmax = np.max(chain_cut, axis=0)
            if diff:
                Q = np.column_stack((sam.p0, sam.p0-vmin, vmax-sam.p0))
            else:
                Q = np.column_stack((sam.p0, vmin, vmax))
        else:
            Q = [np.percentile(par, [50, 16, 84]) for par in sam.chain.T]
            Q = np.column_stack(Q).T
            if diff:
                Q[:, 1] = Q[:, 0] - Q[:, 1]
                Q[:, 2] = Q[:, 2] - Q[:, 0]

        kwargs = lik.build_kwargs(Q)

        # calculate by
        w = kwargs["width"]
        w = w if type(w) is float else w[0]  # for fixed w
        zz, NN = get_dndz(d.tracers[0][0].dndz, w)
        zmean = np.average(zz, weights=NN)
        sigz = np.sqrt(np.sum(NN * (zz - zmean)**2) / np.sum(NN))
        zarr = np.linspace(zmean - sigz, zmean + sigz, 10)
        bys = np.array([hm_bias(cosmo, 1/(1 + zarr), d.tracers[1][1].profile,
                  **(lik.build_kwargs(p0))) for p0 in sam.chain[::100]])
        bymin, by, bymax = np.percentile(bys, [16, 50, 84])
        if diff: bymin=by-bymin; bymax=bymax-by
        kwargs["by"] = [by, bymin, bymax]

        params.append(kwargs)
        chi2.append(lik.chi2(sam.p0))
        dof.append(len(lik.dv))

    return params, (chi2, dof)

