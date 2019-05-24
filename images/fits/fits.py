import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


dir1 = "../../output/"
wisc = ["wisc%d" % i for i in range(1, 6)]
surveys = ["2mpz"] + wisc
sci = [r"$\mathrm{2MPZ}$"] + \
      [r"$\mathrm{WI \times SC}$ - $\mathrm{%d}$" % i for i in range(1, 6)]



f = plt.figure(figsize=(8, 12))
gs_main = GridSpec(6, 2, figure=f)

for s, sur in enumerate(surveys):
    ggyg = [sur, "y_milca"]

    for i, g in enumerate(ggyg):
        # extract data
        fname = dir1+"sampler_minimal_hmc_"+sur+"_cls_"+sur+"_"+g+".npy"
        ll, dd, ee, tt, h1, h2, c2, do = np.load(fname)
        chi2 = c2[0]
        dof = do[0]

        if i == 0:
            xx = 1  # no. of omitted ells
            dname = dir1+"cls_"+sur+"_"+g+".npz"
            ll = np.append(np.load(dname)["ls"][:xx], ll)
            dd = np.append(np.load(dname)["cls"][:xx], dd)

            cname = dir1+"cov_comb_m_"+sur+"_"+g+"_"+sur+"_"+g+".npz"
            ee = np.append(np.sqrt(np.diag(np.load(cname)["cov"])[:xx]), ee)
        else:
            xx = 0


        # set-up subplot
        gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[s, i],
                                     height_ratios=[3, 1], hspace=0)
        ax1 = f.add_subplot(gs[0])
        ax2 = f.add_subplot(gs[1])


        # plot data & theory
        ax1.plot(ll[xx:], h1, ls="-", c="darkgreen", alpha=0.3,
                 label=r"$\mathrm{1}$-$\mathrm{halo}$")
        ax1.plot(ll[xx:], h2, ls="-", c="navy", alpha=0.3,
                 label=r"$\mathrm{2}$-$\mathrm{halo}$")
        ax1.errorbar(ll, dd, yerr=ee, fmt="r.")
        ax1.plot(ll[xx:], tt, "k-", label=r"$\mathrm{1h+2h}$")

        ax2.errorbar(ll[xx:], (dd[xx:]-tt)/ee[xx:],
                     yerr=np.ones_like(dd[xx:]), fmt="r.")
        ax2.plot([ll[0]/1.1, ll[-1]*1.1], [0, 0], "k--")


        # format plot
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")

        ax1.set_xlim([ll[0]/1.1, ll[-1]*1.1])
        ax2.set_xlim([ll[0]/1.1, ll[-1]*1.1])

        if i == 0:
            ax1.axvspan(ax1.get_xlim()[0], ll[xx], color="grey", alpha=0.3)
            ax2.axvspan(ax1.get_xlim()[0], ll[xx], color="grey", alpha=0.4)
            ax1.text(0.02, 0.05, sci[s] + "\n" + \
                     "$\\chi^2/{\\rm dof} = %.2lf / %d$" % (chi2, dof),
                     transform=ax1.transAxes)

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
plt.savefig("fits.pdf", bbox_inches="tight")
