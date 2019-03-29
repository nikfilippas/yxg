import numpy as np


class Likelihood(object):
    """
    Manages parameters (free, fixed, coupled etc.), and computes
    likelihoods for them.

    Args:
        pars (dict): dictionary of parameters, as provided in the
            input YAML file.
        data (array): 1D array with the data vector.
        get_theory (function): function that, given a dictionary of
            parameters, returns the theory prediction for the data
            vector provided in `data`.
        debug (bool): if True, will output debugging information.
    """
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
        """
        Given a list of free parameter values, int constructs the
        dictionary of parameters needed to get a theory prediction.

        Args:
            par (list): list of values for all free parameters.

        Returns:
            dictionary: dictionary of all parameters needed to compute
                the theory prediction.
        """
        params = dict(self.p_fixed)
        params.update(dict(zip(self.p_free_names, par)))
        for p1, p2 in self.p_alias:
            params[p1] = params[p2]
        return params

    def lnprior(self, par):
        """
        Computes the prior for a list of input free parameters.

        Args:
            par (list): list of values for all free parameters.

        Returns:
            float: log prior.
        """
        lnp = 0
        for p, pr in zip(par, self.p_free_prior):
            if pr is None:  # No prior
                continue
            elif pr['type'] == 'Gaussian':
                lnp += -0.5 * ((p - pr['values'][0]) / pr['values'][1])**2
            else:
                if not(pr['values'][0] <= p <= pr['values'][1]):
                    return -np.inf
        return lnp

    def lnlike(self, par):
        """
        Computes the likelihood for a list of input free parameters.

        Args:
            par (list): list of values for all free parameters.

        Returns:
            float: log likelihood.
        """
        params = self.build_kwargs(par)
        tv = self.get_theory(params)
        if tv is None:  # Theory calculation failed
            return -np.inf
        dx = self.dv-tv
        return -0.5 * np.einsum('i,ij,j', dx, self.ic, dx)

    def generate_data(self, par):
        """
        Generates a sample of the data given a list of input
        free parameters.

        Args:
            par (list): list of values for all free parameters.

        Returns:
            array: Gaussian realization of the data vector.
        """
        params = self.build_kwargs(par)
        tv = self.get_theory(params)
        return tv+np.dot(self.cvhalf, np.random.randn(len(tv)))

    def lnprob(self, par):
        """
        Computes the posterior for a list of input free parameters.

        Args:
            par (list): list of values for all free parameters.

        Returns:
            float: log posterior.
        """
        pr = self.lnprior(par)
        if pr != -np.inf:
            pr += self.lnlike(par)

        if self.debug:
            print(par, pr)

        return pr

    def chi2(self, par):
        """
        Computes -2 times the log posterior for a list of input
        free parameters. Useful for minimization.

        Args:
            par (list): list of values for all free parameters.

        Returns:
            float: -2 times the log posterior (~ "chi^2").
        """
        pr = self.lnprior(par)
        if pr != -np.inf:
            pr += self.lnlike(par)

        if self.debug:
            print(par, -2 * pr)

        return -2 * pr

    def plot_data(self, par, dvec, save_figures=False, prefix=None):
        """
        Produces a plot of the different data power spectra with
        error bars and the theory prediction corresponding to a set
        of input free parameters. Figures can be saved to file
        automatically.

        Args:
            par (list): list of values for all free parameters.
            dvec (:obj:`DataManager describing the structure of
                the data vector.
            save_figures (bool): if true, figures will be saved to
                file. File names will take the form:
                <`prefix`>cls_<tracer1>_<tracer2>.png
                where tracer1 and tracer2 are the names of the two
                tracers contributing to a given power spectrum.
            prefix (str): output prefix.

        Returns:
            array of figure objects.
        """
        import matplotlib.pyplot as plt

        params = self.build_kwargs(par)
        # Array of multipoles
        ls = np.array(dvec.ells)
        # Indices used in the analysis
        indices = np.arange(len(ls.flatten()), dtype=int)
        indices = indices.reshape(ls.shape)

        # Compute theory prediction and reshape to
        # [n_correlations,n_ells]
        tv = self.get_theory(params)[indices]
        # Reshape data vector
        dv = self.dv[indices]
        # Compute error bars and reshape
        ev = np.sqrt(np.diag(self.cv))[indices]
        # Compute chi^2
        chi2 = self.chi2(par)
        dof = len(self.dv)

        # Loop through each correlation and produce a figure
        figs = []
        ax = []
        for ll, tt, dd, ee, tr in zip(ls, tv, dv, ev,
                                      dvec.tracers):
            typ_str = ''
            for t in tr:
                typ_str += t.type

            fig = plt.figure()
            ax1 = fig.add_axes((.1, .3, .8, .6))
            # Data with errorbars
            ax1.errorbar(ll, dd, yerr=ee, fmt='r.')
            # Theory
            ax1.plot(ll, tt, 'k-')
            ax1.set_xlabel('$\\ell$', fontsize=15)
            ax1.set_ylabel('$C^{' + typ_str +
                           '}_\\ell$', fontsize=15)
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.set_xlim([ll[0]/1.1, ll[-1]*1.1])
            ax2 = fig.add_axes((.1, .1, .8, .2))
            ax2.set_xlim([ll[0]/1.1, ll[-1]*1.1])
            # Normalized residuals
            ax2.errorbar(ll, (dd - tt) / ee, yerr=np.ones_like(dd), fmt='r.')
            ax2.plot([ll[0]/1.1, ll[-1]*1.1], [0, 0], 'k--')
            ax2.set_xlabel('$\\ell$', fontsize=15)
            ax2.set_ylabel('$\\Delta_\\ell$', fontsize=15)
            ax2.set_xscale('log')
            ax.append(ax1)
            ax.append(ax2)
            figs.append(fig)
        # Print the chi^2 value in the first plot
        ax[0].text(0.7, 0.85,
                   '$\\chi^2/{\\rm dof} = %.2lf / %d$' % (chi2, dof),
                   transform=ax[0].transAxes)

        if save_figures:
            if prefix is None:
                raise ValueError("Need a file prefix to save stuff")
            for fig, tr in zip(figs, dvec.tracers):
                fname = prefix+'cls_'+tr[0].name+'_'+tr[1].name+'.png'
                fig.savefig(fname, bbox_inches='tight')
        return figs

    def plot_chain(self, chain, save_figure=False, prefix=None):
        """
        Produces a triangle plot from a chain, which can be
        saved to file automatically.

        Args:
            chain (array): 2D array with shape [n_samples,n_params],
                where `n_samples` is the number of samples in the
                chain and `n_params` is the number of free parameters
                for this likelihood run.
            save_figures (bool): if true, figures will be saved to
                file. File names will take the form:
                <`prefix`>triangle.png
            prefix (str): output prefix.

        Returns:
            figure object
        """
        from getdist import MCSamples
        from getdist import plots as gplots

        nsamples = len(chain)
        # Generate samples
        samples = MCSamples(samples=chain[nsamples//4:],
                            names=self.p_free_names,
                            labels=self.p_free_labels)
        # Triangle plot
        g = gplots.getSubplotPlotter()
        g.triangle_plot([samples], filled=True)

        if save_figure:
            if prefix is None:
                raise ValueError("Need a file prefix to save stuff")
            fname = prefix+'triangle.png'
            print(fname)
            g.export(fname)
        return g
