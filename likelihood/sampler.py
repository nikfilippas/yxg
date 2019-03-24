import numpy as np


class Sampler(object):
    def __init__(self, lnprob, p0, prefix_out):
        self.lnprob = lnprob
        self.p0 = p0
        self.prefix_out = prefix_out

        def chi2(p):
            return -2 * self.lnprob(p)

        self.chi2 = chi2

    def update_p0(self, p0):
        self.p0 = p0

    def get_best_fit(self, p0=None,
                     xtol=0.0001, ftol=0.0001, maxiter=None,
                     options=None, update_p0=False):
        from scipy.optimize import minimize

        if p0 is None:
            p0 = self.p0

        opt = {'xtol': xtol, 'ftol': ftol, 'maxiter': maxiter}
        if options is not None:
            opt.update(options)

        res = minimize(self.chi2, p0, method="Powell", options=opt)

        if update_p0:
            self.update_p0(res.x)

        return res.x

    def get_covariance(self, p0=None):
        import numdifftools as nd

        if p0 is None:
            p0 = self.p0

        invcov = -nd.Hessian(self.lnprob)(p0)
        try:
            cov = np.linalg.inv(invcov)
        except np.linalg.linalg.LinAlgError:
            cov = np.diag(1./np.fabs(np.diag(invcov)))

        return cov
