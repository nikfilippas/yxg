import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def gaus(x, mu, sigma):
    """Returns a normalised gaussian with (mu, sigma) parameters."""
    a = 1/(sigma*np.sqrt(2*np.pi))
    p = a*np.exp(-0.5*((x-mu)/sigma)**2)
    p /= simps(p, x)
    return p


def modgaus(z, nz, w):
    """Modified Gaussian function."""
    nzf = interp1d(z, nz, kind="cubic", bounds_error=False, fill_value=0)
    z_avg = np.average(z, weights=nz)
    nz_new = nzf(z_avg + (1/w)*(z-z_avg))
    nz_new /= simps(nz_new, x=z)
    return nz_new


def convolve(z, a1, a2):
    """Convolves and normalises two arrays."""
    nz = np.correlate(a1, a2, "full")[::2]
    nz /= simps(nz, x=z)
    return nz


def nz_maker(z, f1=(0.20, 0.25), f2=(0.00, 0.04), full_output=False):
    """Makes fiducial N(z)'s for a top-hat + Gaussian combination."""
    # tophat
    top = np.zeros_like(z)
    ones = np.where((z >= f1[0]) & (z <= f1[1]))[0]
    top[ones] = 1
    top /= simps(top, x=z)
    # Gaussian
    pr = gaus(z, f2[0], f2[1])
    # convolve & normalise
    nz = convolve(z, top, pr)
    if full_output:
        return top, pr, nz
    else:
        return nz



# redshifts
z = np.linspace(-0.3, 0.50, 1000)
s_fid = 0.04
s_ext = 1.50
sigmas = np.append(np.linspace(0.99, 1.01, 100), s_ext)*s_fid

top, pr, nz_fid = nz_maker(z, full_output=True)

nz = np.array([nz_maker(z, f2=(0.00, s), full_output=False) for s in sigmas])

p = [curve_fit(lambda z, w: modgaus(z, nz_fid, w), z, N) for N in nz]
p = np.array(p).flatten().reshape((len(p), 2))
wopt, wcov = p.T

soff = 100*(sigmas-s_fid)/s_fid
woff = 100*(wopt-1)

nz_fit = modgaus(z, nz_fid, wopt[-1])



plt.figure()
plt.plot(soff[:-1], woff[:-1], "k", lw=3)
plt.xlabel("percentage offset in $\sigma$")
plt.ylabel("percentage offset in $w$")
plt.savefig("error_test1.pdf")

plt.figure()
plt.plot(z, top, "k:", label="$p(z_{photo})$")
plt.plot(z, pr, "k--", label="$p(z|z_{photo})$")
plt.plot(z, nz_fid, "k-", lw=4, label="$p_{fid} = N(z|0, \sigma_{fid})$")
plt.plot(z, nz[-1], "r-", lw=2, label="$N(z|0, %.1f\sigma_{fid})$" % s_ext)
plt.plot(z, nz_fit, "b--", lw=1, label="best fit")
plt.legend()
plt.savefig("error_test2.pdf")