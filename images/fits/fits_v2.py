import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


yamlfile = "../../params.yml"
dir1 = "../../output/"
wisc = ["wisc%d" % i for i in range(1, 6)]
surveys = ["2mpz"] + wisc
sci = [r"$\mathrm{2MPZ}$"] + \
      [r"$\mathrm{WI \times SC}$ - $\mathrm{%d}$" % i for i in range(1, 6)]


with open(yamlfile) as f:
    P = yaml.safe_load(f)["data_vectors"]

f = plt.figure(figsize=(8, 12))
gs_main = GridSpec(6, 2, figure=f)

for s, sur in enumerate(surveys):
    ggyg = [sur, "y_milca"]

    for i, g in enumerate(ggyg):
        ## extract parameters ##
        fname = dir1+"best_fit_params_"+sur+".npy"
        par = np.load(fname)
        params, (chi2, dof) = par[:-2], par[-2:]

        # data
        dname = dir1+"cls_"+g+"_"+sur+".npz"
        ll = np.load(dname)["ls"]
        dd = np.load(dname)["cls"]

        cname = dir1+"cov_comb_m_"+sur+"_"+g+"_"+sur+"_"+g+".npz"
        ee = np.sqrt(np.diag(np.load(cname)["cov"]))

        # theory # TODO: put correct formulas in
        h1 = np.ones_like(ll)
        h2 = 2*np.ones_like(ll)
        tt = 3*np.ones_like(ll)

        # used indices
        lmin = P[s]["twopoints"][i]["lmin"]
        lmax = 1000  # TODO: calculate it
        ind0 = np.where((lmax > ll) & (ll > lmin))[0]
        ind1 = np.where((lmax > ll))[0]
        ind2 = np.hstack((ind1, ind0))

        # set-up subplot
        gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[s, i],
                                     height_ratios=[3, 1], hspace=0)
        ax1 = f.add_subplot(gs[0])
        ax2 = f.add_subplot(gs[1])



        # plot data & theory
        ax1.plot(ll[ind0], h1[ind0], ls="-", c="darkgreen", alpha=0.3,
                 label=r"$\mathrm{1}$-$\mathrm{halo}$")
        ax1.plot(ll[ind1], h1[ind1], ls=":", c="darkgreen", alpha=0.3)
        ax1.plot(ll[ind0], h2[ind0], ls="-", c="navy", alpha=0.3,
                 label=r"$\mathrm{2}$-$\mathrm{halo}$")
        ax1.plot(ll[ind1], h2[ind1], ls=":", c="navy", alpha=0.3)

        ax1.errorbar(ll[ind2], dd[ind2], yerr=ee[ind2], fmt="r.")

        ax1.plot(ll[ind0], tt[ind0], "k-", label=r"$\mathrm{1h+2h}$")
        ax1.plot(ll[ind1], tt[ind1], "k:")

        ax2.errorbar(ll[ind2], (dd[ind2]-tt[ind2])/ee[ind2],
                     yerr=np.ones_like(dd[ind2]), fmt="r.")
        ax2.axhline(color="k", ls="--")


        # format plot
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")

        ax1.set_xlim([ll[ind2][0]/1.1, ll[ind2][-1]*1.1])
        ax2.set_xlim([ll[ind2][0]/1.1, ll[ind2][-1]*1.1])

        if i == 0:
            ax1.axvspan(ax1.get_xlim()[0], ll[ind0][0], color="grey", alpha=0.2)
            ax2.axvspan(ax1.get_xlim()[0], ll[ind0][0], color="grey", alpha=0.2)
            ax1.text(0.02, 0.05, sci[s]+"\n"+"$\\chi^2/{\\rm dof}=%.2lf/%d$" %
                     (chi2, dof), transform=ax1.transAxes)

            ax1.set_ylabel('$C_\\ell$', fontsize=15)
            ax2.set_ylabel('$\\Delta_\\ell$', fontsize=15)

        if s == 0:
            if i == 0:
                ax1.text(0.45, 1.1, r"$g \times g$", fontsize=15,
                         transform=ax1.transAxes)
            if i == 1:
                ax1.text(0.45, 1.1, r"$y \times g$", fontsize=15,
                         transform=ax1.transAxes)
                ax1.legend(loc="upper center", ncol=3, frameon=False,
                           bbox_to_anchor=(0.52, 1.05))

        if s != len(surveys)-1:
            ax2.get_xaxis().set_visible(False)
        else:
            ax2.set_xlabel('$\\ell$', fontsize=15)


plt.tight_layout(h_pad=0.05, w_pad=0.1)
#plt.savefig("fits.pdf", bbox_inches="tight")
