import numpy as np
import os
import warnings


class Sampler(object):
    def __init__(self, lnprob, p0, parnames, prefix_out, par,
                 covar=None, chain=None):
        self.lnprob = lnprob
        self.ndim = len(p0)
        self.parnames = parnames
        self.p0 = p0
        self.prefix_out = prefix_out
        self.nwalkers = par['n_walkers']
        self.nsteps = par['n_steps']
        self.covar = None
        self.chain = None
        self.probs = None

        def chi2(p):
            return -2 * self.lnprob(p)

        self.chi2 = chi2

    def update_p0(self, p0):
        self.p0 = p0

    def update_cov(self, cov):
        self.covar = cov

    def read_properties(self):
        fname_props = self.prefix_out + "properties.npz"

        if os.path.isfile(fname_props):
            with np.load(fname_props) as f:
                self.update_p0(f['p0'])
                self.update_cov(f['covar'])
            return True
        return False

    def get_best_fit(self, p0=None,
                     xtol=0.0001, ftol=0.0001, maxiter=None,
                     options=None, update_p0=False):
        from scipy.optimize import minimize

        if p0 is None:
            p0 = self.p0

        opt = {'xtol': xtol, 'ftol': ftol, 'maxiter': maxiter}
        if options is not None:
            opt.update(options)

        with warnings.catch_warnings():  # Suppress warnings due to np.inf
            warnings.simplefilter("ignore")
            res = minimize(self.chi2, p0, method="Powell", options=opt)

        if update_p0:
            self.update_p0(res.x)

        return res.x

    def get_covariance(self, p0=None, update_cov=False):
        import numdifftools as nd

        if p0 is None:
            p0 = self.p0

        with warnings.catch_warnings():  # Suppress warnings due to np.inf
            warnings.simplefilter("ignore")
            invcov = -nd.Hessian(self.lnprob)(p0)

        try:
            cov = np.linalg.inv(invcov)
        except np.linalg.linalg.LinAlgError:
            cov = np.diag(1./np.fabs(np.diag(invcov)))

        if update_cov:
            self.update_cov(cov)
        return cov

    def sample(self, carry_on=False, verbosity=0):
        import emcee
        from multiprocessing import Pool

        fname_chain = self.prefix_out+"chain.h5"

        backend = emcee.backends.HDFBackend(fname_chain)

        found_file = os.path.isfile(fname_chain)
        if (not found_file) or (not carry_on):
            backend.reset(self.nwalkers, self.ndim)
            pos = (np.array(self.p0)[None, :] +
                   0.001 * np.random.randn(self.nwalkers, self.ndim))
            nsteps_use = self.nsteps
        else:
            print("Restarting from previous run")
            pos = None
            nsteps_use = max(self.nsteps-len(backend.get_chain()), 0)

        # Multiprocessing
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(self.nwalkers,
                                            self.ndim,
                                            self.lnprob,
                                            backend=backend,
                                            pool=pool)
            if nsteps_use > 0:
                sampler.run_mcmc(pos, nsteps_use, store=True,
                                 progress=(verbosity > 0))

        return sampler

    def get_chain(self):
        import emcee

        fname_chain = self.prefix_out + "chain.h5"
        reader = emcee.backends.HDFBackend(fname_chain, read_only=True)
        self.chain = reader.get_chain(flat=True)
        self.probs = reader.get_log_prob(flat=True)

    def save_properties(self):
        fname_props = self.prefix_out + "properties"
        np.savez(fname_props, names=self.parnames,
                 p0=self.p0, covar=self.covar,
                 nwalkers=self.nwalkers, nsteps=self.nsteps)
