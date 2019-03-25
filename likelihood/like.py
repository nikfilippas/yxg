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

    def plot_data(self, par, dvec):
        import matplotlib.pyplot as plt

        params = self.build_kwargs(par)
        ls = np.array(dvec.ells)
        indices = np.arange(len(ls.flatten()), dtype=int)
        indices = indices.reshape(ls.shape)

        tv = self.get_theory(params)[indices]
        dv = self.dv[indices]
        ev = np.sqrt(np.diag(self.cv))[indices]
        chi2 = self.chi2(par)
        dof = len(self.dv)

        figs = []
        ax = []
        for ll, tt, dd, ee, tr in zip(ls, tv, dv, ev,
                                      dvec.tracers):
            typ_str = ''
            for t in tr:
                typ_str += t.type

            fig = plt.figure()
            ax1 = fig.add_axes((.1, .3, .8, .6))
            ax1.errorbar(ll, dd, yerr=ee, fmt='r.')
            ax1.plot(ll, tt, 'k-')
            ax1.set_xlabel('$\\ell$', fontsize=15)
            ax1.set_ylabel('$C^{' + typ_str +
                           '}_\\ell$', fontsize=15)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlim([ll[0]/1.1, ll[-1]*1.1])
            ax2 = fig.add_axes((.1, .1, .8, .2))
            ax2.set_xlim([ll[0]/1.1, ll[-1]*1.1])
            ax2.errorbar(ll, (dd - tt) / ee, yerr=np.ones_like(dd), fmt='r.')
            ax2.plot([ll[0]/1.1, ll[-1]*1.1], [0, 0], 'k--')
            ax2.set_xlabel('$\\ell$', fontsize=15)
            ax2.set_ylabel('$\\Delta_\\ell$', fontsize=15)
            ax2.set_xscale('log')
            ax.append(ax1)
            ax.append(ax2)
            figs.append(fig)
        ax[0].text(0.7, 0.85,
                   '$\\chi^2/{\\rm dof} = %.2lf / %d$' % (chi2, dof),
                   transform=ax[0].transAxes)
        return figs

    def plot_chain(self, chain):
        from getdist import MCSamples
        from getdist import plots as gplots

        nsamples = len(chain)
        samples = MCSamples(samples=chain[nsamples//4:],
                            names=self.p_free_names,
                            labels=self.p_free_labels)
        g = gplots.getSubplotPlotter()
        g.triangle_plot([samples], filled=True)

        print(dir(g))
        return g
