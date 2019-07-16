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


def get_dndz(fname, width):
    """Get the modified galaxy number counts."""
    zd, Nd = np.loadtxt(fname, unpack=True)
    Nd /= simps(Nd, x=zd)
    zavg = np.average(zd, weights=Nd)
    nzf = interp1d(zd, Nd, kind="cubic", bounds_error=False, fill_value=0)

    Nd_new = nzf(zavg + (1/width)*(zd-zavg))
    return zd, Nd_new


def vpercentile(chain, percentile=68, bins=100, eps=0.005):
    """Finds the parameter errors using a Markov chain."""
    pdf, x = np.histogram(chain, bins, density=True)

    cut = pdf.max()*np.flip(np.arange(eps, 1, eps))
    for cc in cut:

        bb = np.where(pdf-cc > 0)[0]
        if bb.size < 2: continue
        par_min, par_max = x[bb[0]], x[bb[-1]]
        N_enclosed = (par_min < chain) & (chain < par_max)
        perc = 100*N_enclosed.sum()/chain.size
        if perc > percentile: break

    return (par_min, par_max)


def sampler2kwargs(sam, lik, diff=False, error_type=None):
    """Builds kwargs given a sampler."""

    if error_type == "probability":
        sam.update_p0(sam.chain[np.argmax(sam.probs)])
        P_cut = np.percentile(sam.probs, 68)
        chain_cut = sam.chain[sam.probs > P_cut]
        vmin = np.min(chain_cut, axis=0)
        vmax = np.max(chain_cut, axis=0)
        if diff:
            Q = np.column_stack((sam.p0, sam.p0-vmin, vmax-sam.p0)).T
        else:
            Q = np.column_stack((sam.p0, vmin, vmax)).T
    elif error_type == "hpercentile":
        Q = [np.percentile(par, [50, 16, 84]) for par in sam.chain.T]
        Q = np.vstack(Q).T
        if diff:
            Q[1] = Q[0] - Q[1]
            Q[2] = Q[2] - Q[0]
    elif error_type == "vpercentile":
        Q = np.vstack([vpercentile(chain) for chain in sam.chain.T]).T
        Q = np.vstack((sam.p0, Q))
        # fix maxima/minima
        Q[1] = np.min([Q[0], Q[1]], axis=0)
        Q[2] = np.max([Q[0], Q[2]], axis=0)
        if diff:
            Q[1] = Q[0] - Q[1]
            Q[2] = Q[2] - Q[0]
    else:
        raise ValueError("Provide an error type!")

    kwargs = lik.build_kwargs(Q.T)

    return kwargs


def chan(fname_params, diff=False, error_type=None, b_hydro=None, chains=True):
    """
    Given a parameter file, looks up corresponding sampler and calculates
    best-fit parameters, and min & max values.
    Outputs a dictionary of best-fit parameters, chi-squared and dof.

    Args:
        fname_params (str): Name of parameter file.
        diff (bool): If True, return differences `vv-vmin`, `vmax-vv`.
                     If False, return min and max values.
        error_type (str): How to estimate errors:
                        - `probability`: 68% of highest probability samples
                        - `hpercentile`: 68% using 16-84 percentiles
                        - `vpercentile`: 68% using watershed method
        b_hydro (float): Custom `b_hydro` value. If `None`, use `p0`.
        chains (bool): Whether to return chains.

    Returns:
        params (dict): Dictionary of values. "name" : [50, 16, 84] probability-
                       wise percentiles.
        chi2, dof (float): Chi-squared of best-fit and degrees of freedom.
        chains: The chains of the fitted parameters.
    """
    def th(pars):
        return get_theory(p, d, cosmo, hm_correction=hm_correction,
                          selection=sel, **pars)

    p = ParamRun(fname_params)
    cosmo = p.get_cosmo()
    sel = selfunc(p)
    hm_correction = halmodcor(p, cosmo)

    params, chi2, dof = [[] for i in range(3)]
    chains = [[],[]]
    for s, v in enumerate(p.get("data_vectors")):

        # Construct data vector and covariance
        d = DataManager(p, v, cosmo, all_data=False)
        lik = Likelihood(p.get('params'), d.data_vector, d.covar, th)
        sam = Sampler(lik.lnprob, lik.p0, lik.p_free_names,
                      p.get_sampler_prefix(v['name']), p.get('mcmc'))

        # Read chains and best-fit
        sam.get_chain()
        sam.update_p0(sam.chain[np.argmax(sam.probs)])

        kwargs = sampler2kwargs(sam, lik, diff=diff, error_type=error_type)

        # calculate b_y
        w = kwargs["width"]
        w = w if type(w) is float else w[0]  # for fixed w
        zz, NN = get_dndz(d.tracers[0][0].dndz, w)
        zmean = np.average(zz, weights=NN)
        sigz = np.sqrt(np.sum(NN * (zz - zmean)**2) / np.sum(NN))
        zarr = np.linspace(zmean - sigz, zmean + sigz, 10)

        # p0 or user-input b_hydro
        if b_hydro is None:
            bys = np.array([hm_bias(cosmo, 1/(1 + zarr),
                        d.tracers[1][1].profile,
                        **(lik.build_kwargs(p0))) for p0 in sam.chain[::100]])
            bymin, by, bymax = np.percentile(bys, [16, 50, 84])
            if diff: bymin=by-bymin; bymax=bymax-by
            kwargs["by"] = [by, bymin, bymax]
        else:
            bys = np.array([hm_bias(cosmo, 1/(1 + zarr),
                            d.tracers[1][1].profile,
                            **{"b_hydro": bh}) for bh in b_hydro[:, s]])
            kwargs["by"] = [by.mean() for by in bys]

        kwargs["z"] = zmean

        params.append(kwargs)
        chi2.append(lik.chi2(sam.p0))
        dof.append(len(lik.dv))
        if chains:
            chains[0].append(sam.chain)
            chains[1].append(bys.flatten())

    return (params, (chi2, dof), chains) if chains else (params, (chi2, dof))
