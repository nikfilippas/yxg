import numpy as np


class Likelihood(object):
    def __init__(self, pars, data, covar, get_theory, debug=False):
        self.p_free_names = []
        self.p_free_labels = []
        self.p_free_prior = []
        self.p_fixed = []
        self.p_alias = []
        self.p0 = []
        self.get_theory = get_theory
        self.dv = data
        self.cv = covar
        self.ic = np.linalg.inv(covar)
        self.cvhalf = np.linalg.cholesky(covar)
        self.debug = debug

        for p in pars:
            n = p.get('name')
            if p.get('alias') is not None:
                self.p_alias.append((n, p.get('alias')))
            elif not p['vary']:
                self.p_fixed.append((n, p.get('value')))
            else:
                self.p_free_names.append(n)
                self.p_free_labels.append(p.get('label'))
                self.p_free_prior.append(p.get('prior'))
                self.p0.append(p.get('value'))

    def build_kwargs(self, par):
        params = dict(self.p_fixed)
        params.update(dict(zip(self.p_free_names, par)))
        for p1, p2 in self.p_alias:
            params[p1] = params[p2]
        return params

    def lnprior(self, par):
        lnp = 0
        for p, pr in zip(par, self.p_free_prior):
            if pr is None:  # No prior
                continue
            elif pr['type'] == 'Gaussian':
                lnp += ((p - pr['values'][0]) / pr['values'][1])**2
            else:
                if not(pr['values'][0] <= p <= pr['values'][1]):
                    return -np.inf
        return lnp

    def lnlike(self, par):
        params = self.build_kwargs(par)
        tv = self.get_theory(params)
        if tv is None:  # Theory calculation failed
            return -np.inf
        dx = self.dv-tv
        return -0.5 * np.einsum('i,ij,j', dx, self.ic, dx)

    def generate_data(self, par):
        params = self.build_kwargs(par)
        tv = self.get_theory(params)
        return tv+np.dot(self.cvhalf, np.random.randn(len(tv)))

    def lnprob(self, par):
        pr = self.lnprior(par)
        if pr != -np.inf:
            pr += self.lnlike(par)

        if self.debug:
            print(par, pr)

        return pr

    def chi2(self, par):
        pr = self.lnprior(par)
        if pr != -np.inf:
            pr += self.lnlike(par)

        if self.debug:
            print(par, -2 * pr)

        return -2 * pr
