import sys
sys.path.insert(0, '/home/koukoufilippasn/Desktop/DPhil/yxg/model')
import profile2D as p2D
import power_spectrum as pspec
import yaml
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


yamlfile = "../../params.yml"
run_name = "run_fiducial"
dir1 = "../../output/"
wisc = ["wisc%d" % i for i in range(1, 6)]
surveys = ["2mpz"] + wisc
sci = [r"$\mathrm{2MPZ}$"] + \
      [r"$\mathrm{WI \times SC}$ - $\mathrm{%d}$" % i for i in range(1, 6)]


with open(yamlfile) as f:
    P = yaml.safe_load(f)

mf = P["mcmc"]["mfunc"]
cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665,
                      mass_function=mf)


f = plt.figure(figsize=(8, 12))
gs_main = GridSpec(6, 2, figure=f)

for s, sur in enumerate(surveys):
    ggyg = [sur, "y_milca"]

    for i, g in enumerate(ggyg):
        # data
        dname = dir1+"cls_"+g+"_"+sur+".npz"
        ll = np.load(dname)["ls"]
        dd = np.load(dname)["cls"]

        cname = dir1+"cov_comb_m_"+sur+"_"+g+"_"+sur+"_"+g+".npz"
        ee = np.sqrt(np.diag(np.load(cname)["cov"]))

        # used indices
        lmin = P["data_vectors"][s]["twopoints"][i]["lmin"]

        Nz_file = P["maps"][s]["dndz"]
        Nz_data = np.loadtxt("../../"+Nz_file)
        z_avg = np.average(Nz_data[:, 0], weights=Nz_data[:, 1])

        chi = ccl.comoving_radial_distance(cosmo, 1/(1+z_avg))
        kmax = P["mcmc"]["kmax"]
        lmax = kmax*chi - 0.5

        ind0 = np.where((lmax > ll) & (ll > lmin))[0]  # fitted

        # theory
        p2 = p2D.HOD(nz_file="../../"+Nz_file)
        if i == 1:
            p1 = p2D.Arnaud()
        else:
            p1 = p2
        prof = (p1, p2)

        fname = dir1+"best_fit_params_"+run_name+"_"+sur+".npy"
        par = np.load(fname)
        params, (chi2, dof) = par[:-2], par[-2:]
        kwargs = {"M0"         :  params[1],
                  "M1"         :  params[0],
                  "Mmin"       :  params[1],
                  "alpha"      :  1.0,
                  "b_hydro"    :  params[2],
                  "beta_gal"   :  1.0,
                  "beta_max"   :  1.0,
                  "fc"         :  1.0,
                  "r_corr"     :  params[3],
                  "sigma_lnM"  :  0.15}

        h1 = pspec.hm_ang_power_spectrum(cosmo, ll, prof,
                                include1h=True, include2h=False,
                                hm_correction=pspec.HalomodCorrection(cosmo),
                                **kwargs)
        h2 = pspec.hm_ang_power_spectrum(cosmo, ll, prof,
                                include1h=False, include2h=True,
                                hm_correction=pspec.HalomodCorrection(cosmo),
                                **kwargs)
        tt = pspec.hm_ang_power_spectrum(cosmo, ll, prof,
                                include1h=True, include2h=True,
                                hm_correction=pspec.HalomodCorrection(cosmo),
                                **kwargs)


        # set-up subplot
        gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[s, i],
                                     height_ratios=[3, 1], hspace=0)
        ax1 = f.add_subplot(gs[0])
        ax2 = f.add_subplot(gs[1])

        # plot data & theory
        ax1.plot(ll[ind0], h1[ind0], ls="-", c="darkgreen", alpha=0.3,
                 label=r"$\mathrm{1}$-$\mathrm{halo}$")
        ax1.plot(ll[:ind0[0]+1], h1[:ind0[0]+1], ls=":", c="darkgreen", alpha=0.3)
        ax1.plot(ll[ind0[-1]:], h1[ind0[-1]:], ls=":", c="darkgreen", alpha=0.3)
        ax1.plot(ll[ind0], h2[ind0], ls="-", c="navy", alpha=0.3,
                 label=r"$\mathrm{2}$-$\mathrm{halo}$")
        ax1.plot(ll[:ind0[0]+1], h2[:ind0[0]+1], ls=":", c="navy", alpha=0.3)
        ax1.plot(ll[ind0[-1]:], h2[ind0[-1]:], ls=":", c="navy", alpha=0.3)

        ax1.errorbar(ll, dd, yerr=ee, fmt="r.")

        ax1.plot(ll[ind0], tt[ind0], "k-", label=r"$\mathrm{1h+2h}$")
        ax1.plot(ll[:ind0[0]+1], tt[:ind0[0]+1], "k:")
        ax1.plot(ll[ind0[-1]:], tt[ind0[-1]:], "k:")

        res = (dd-tt)/ee
        ax2.errorbar(ll, res, yerr=np.ones_like(dd), fmt="r.")
        ax2.axhline(color="k", ls="--")

        # format plot
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")

        ax1.set_xlim(ll[0]/1.1, ll[-1]*1.1)
        ax2.set_xlim(ll[0]/1.1, ll[-1]*1.1)
        ax2.set_ylim(res[ind0].min()/1.1, res[ind0].max()*1.1)

        # grey boundaries
        ax1.axvspan(0.5*(ll[ind0[-1]]+ll[ind0[-1]+1]), ax1.get_xlim()[1],
                    color="grey", alpha=0.2)
        ax2.axvspan(0.5*(ll[ind0[-1]]+ll[ind0[-1]+1]), ax1.get_xlim()[1],
                    color="grey", alpha=0.2)


        if i == 0:
            if lmin != 0:
                ax1.axvspan(ax1.get_xlim()[0], 0.5*(ll[ind0][0]+ll[ind0[0]-1]),
                            color="grey", alpha=0.2)
                ax2.axvspan(ax1.get_xlim()[0], 0.5*(ll[ind0][0]+ll[ind0[0]-1]),
                            color="grey", alpha=0.2)
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
plt.show()
#plt.savefig("fits.pdf", bbox_inches="tight")