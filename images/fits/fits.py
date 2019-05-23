import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


dir1 = "../../output/"
prefix = "sampler_minimal_hmc_"
wisc = ["wisc%d" % i for i in range(1, 6)]
surveys = ["2mpz"] + wisc
sci = [r"$\mathrm{2MPZ}$"] + \
      [r"$\mathrm{WI \times SC - %d}$" % i for i in range(1, 6)]



f = plt.figure(figsize=(8, 12))
gs_main = GridSpec(6, 2, figure=f)

for s, sur in enumerate(surveys):
    ggyg = [sur, "y_milca"]

    for i, g in enumerate(ggyg):
        # extract data
        fname = dir1 + prefix + sur + "_cls_" + sur + "_" + g + ".npy"
        ll, dd, ee, tt, h1, h2, c2, do = np.load(fname)
        chi2 = c2[0]
        dof = do[0]


        # set-up subplot
        gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[s, i],
                                     height_ratios=[3, 1], hspace=0)
        ax1 = f.add_subplot(gs[0])
        ax2 = f.add_subplot(gs[1])


        # plot data & theory
        ax1.errorbar(ll, dd, yerr=ee, fmt="r.")
        ax1.plot(ll, tt, "k-")
        ax1.plot(ll, h1, ls="-", c="darkgreen", alpha=0.3)
        ax1.plot(ll, h2, ls="-", c="navy", alpha=0.3)

        ax2.errorbar(ll, (dd - tt) / ee, yerr=np.ones_like(dd), fmt="r.")
        ax2.plot([ll[0]/1.1, ll[-1]*1.1], [0, 0], "k--")


        # format plot
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")

        ax1.set_xlim([ll[0]/1.1, ll[-1]*1.1])
        ax2.set_xlim([ll[0]/1.1, ll[-1]*1.1])

        if i == 0:
            ax1.text(0.6, 0.75, sci[s] + "\n" + \
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

        if s != len(surveys)-1:
            ax2.get_xaxis().set_visible(False)
        else:
            ax2.set_xlabel('$\\ell$', fontsize=15)


plt.tight_layout(h_pad=0.05, w_pad=0.1)
plt.savefig("fits.pdf", bbox_inches="tight")
