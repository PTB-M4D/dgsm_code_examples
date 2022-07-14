import numpy as np
from PyDynamic.model_estimation import LSIIR
from PyDynamic.uncertainty.propagate_convolution import convolve_unc
from scipy.signal import (cont2discrete, dstep, freqs, freqz, step, impulse, dimpulse)
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag

def MC_cont2discrete(b, a, Uab, dt, runs=500):
    ab = np.hstack((a[1:], b))
    Na_conti = len(a) - 1
    results = []
    AB = multivariate_normal(ab, Uab, allow_singular=True).rvs(runs)
    for ab_tmp in AB:
        a_tmp = np.hstack(([1.0], ab_tmp[:Na_conti]))
        b_tmp = ab_tmp[Na_conti:]
        b_discrete, a_discrete, _ = cont2discrete((b_tmp, a_tmp), dt=dt)
        results.append(np.hstack((a_discrete[1:], np.squeeze(b_discrete))))

    Na_discrete = a_discrete.size - 1
    ab_discrete = np.mean(results, axis=0)
    a_discrete = np.hstack(([1.0], ab_discrete[:Na_discrete]))
    b_discrete = ab_discrete[Na_discrete:]
    uab_discrete = np.cov(np.array(results).T)

    return b_discrete, a_discrete, uab_discrete


def MC_step(b, a, Uab, runs=500, dt=None, use_t=None):
    ab = np.hstack((a[1:], b))
    Na = len(a) - 1
    t = use_t
    h_list = []
    AB = multivariate_normal(ab, Uab, allow_singular=True).rvs(runs)
    for ab_tmp in AB:
        a_tmp = np.hstack(([1.0], ab_tmp[:Na]))
        b_tmp = ab_tmp[Na:]
        if dt is None:
            t, h_tmp = step((b_tmp, a_tmp), T=t)
        else:
            t, h_tmp = dstep((b_tmp, a_tmp, dt), t=t)
            h_tmp = np.squeeze(h_tmp)
        h_list.append(h_tmp)

    h = np.mean(h_list, axis=0)
    uh = np.cov(np.array(h_list).T)
    uh_diag = np.sqrt(np.diag(uh))

    return t, h, uh_diag


def MC_impulse(b, a, Uab, runs=500, dt=None, use_t=None):
    ab = np.hstack((a[1:], b))
    Na = len(a) - 1
    t = use_t
    h_list = []
    AB = multivariate_normal(ab, Uab, allow_singular=True).rvs(runs)
    for ab_tmp in AB:
        a_tmp = np.hstack(([1.0], ab_tmp[:Na]))
        b_tmp = ab_tmp[Na:]
        if dt is None:
            t, h_tmp = impulse((b_tmp, a_tmp), T=t)
        else:
            t, h_tmp = dimpulse((b_tmp, a_tmp, dt), t=t)
            h_tmp = np.squeeze(h_tmp)
        h_list.append(h_tmp)

    h = np.mean(h_list, axis=0)
    uh = np.cov(np.array(h_list).T)
    uh_diag = np.sqrt(np.diag(uh))

    return t, h, uh_diag


def MC_freq(b, a, Uab, dt=None, runs=500, use_w=None, return_full_cov=False):
    W = use_w
    ab = np.hstack((a[1:], b))
    Na = len(a) - 1
    H_list = []

    AB = multivariate_normal(ab, Uab, allow_singular=True).rvs(runs)
    for ab_tmp in AB:
        a_tmp = np.hstack(([1.0], ab_tmp[:Na]))
        b_tmp = ab_tmp[Na:]
        if dt is None:
            W, H_tmp = freqs(b_tmp, a_tmp, worN=W)
        else:
            W, H_tmp = freqz(b_tmp, a_tmp, worN=W, fs=2 * np.pi / dt)
            H_tmp = np.squeeze(H_tmp)
        H_list.append(H_tmp)

    HH = np.array(H_list)
    H = np.mean(HH, axis=0)
    UH = np.cov(np.hstack((np.real(HH), np.imag(HH))).T)

    if return_full_cov:
        return W, H, UH
    else:
        UH_diag = np.sqrt(np.diag(UH)[: len(W)] + np.diag(UH)[len(W) :])
        return W, H, UH_diag


def MC_inverse_iir(b, a, Uab, runs=500, dt_input=None, dt_IIR=1.0, tau=0.0):
    # this seems not to work currently
    # generate frequency response with uncertainty via Monte Carlo
    W, H, UH = MC_freq(b, a, Uab, dt=dt_input, return_full_cov=True)
    Na = len(a[1:])
    Nb = len(b)

    b_inv, a_inv, tau_inv, Uab_inv = LSIIR(
        H, UH=UH, Nb=Na+2, Na=Nb, f=W / (2*np.pi), Fs=1.0 / dt_IIR, tau=tau, inv=True, mc_runs=runs
    )

    # gain correction
    gc = np.sum(a_inv) / np.sum(b_inv)
    b_inv = gc * b_inv
    Uab_inv[Na:,Na:] = Uab_inv[Na:,Na:] * gc**2

    return b_inv, a_inv, Uab_inv


def chain_filter_with_lowpass(b, a, Uab, b_low, a_low):

    # define shortcuts
    Na = len(a) - 1

    uaa = block_diag([0.0], Uab[:Na,:Na])
    ubb = Uab[Na:,Na:]

    bc, Ubc = convolve_unc(b, ubb, b_low, None, mode="full")
    ac, Uac = convolve_unc(a, uaa, a_low, None, mode="full")
    UC = block_diag(Uac[1:,1:], Ubc)

    return bc, ac, UC