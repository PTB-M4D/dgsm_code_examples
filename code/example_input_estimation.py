import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PyDynamic.misc.tools import real_imag_2_complex
from PyDynamic.model_estimation import LSIIR
from PyDynamic.uncertainty.propagate_filter import IIRuncFilter
from scipy.signal import chirp, freqs, lsim, cheby2

from helper_methods import MC_freq, MC_inverse_iir, chain_filter_with_lowpass

###############################
### calculation part ##########
###############################

# true device under test (DUT) transfer behavior ( taken from (*@ \cref{sec:dynunc_calibration_example} @*) )
a_true = np.array([1.00e00, 3.07e03, 3.41e10])
b_true = np.array([3.36e10])

# estimate of DUT parameters ( taken from (*@ \cref{sec:dynunc_models_example} @*) )
a = np.array([1.00e00, 3.77e03, 3.55e10])
b = np.array([3.55e10])
Uab = np.array(
    [
        [1.42e05, 6.11e09, 6.11e09],
        [6.11e09, 1.15e17, 1.15e17],
        [6.11e09, 1.15e17, 2.41e17],
    ]
)

# estimated DUT inverse behavior
fs = 500e3  # Hz
b_inv, a_inv, Uab_inv = MC_inverse_iir(b, a, Uab, dt_input=None, dt_IIR=1 / fs)

# lowpass regularization
b_low, a_low = cheby2(4, 30, 100e3, btype="low", analog=False, output="ba", fs=fs)
b_reg, a_reg, Uab_reg = chain_filter_with_lowpass(b_inv, a_inv, Uab_inv, b_low, a_low)

# acceleration input
t = np.arange(0, 0.01, 1 / fs)
acc_true = chirp(t, f0=10, t1=t[-1], f1=40e3, method="linear", phi=-90)

# measured DUT signal
t, acc_dut, _ = lsim((b_true, a_true), U=acc_true, T=t)
Uacc_dut = np.full_like(
    t, 0.2
)  # assumed uncertainty of indiciation, i.e. known from a calibration

# compensated DUT signal
acc_comp, Uacc_comp, _ = IIRuncFilter(
    acc_dut, Uacc_dut, b_reg, a_reg, Uab_reg, kind="diag"
)


###############################
### visualization part ########
###############################
save_figures = True

# calculate frequency responses (only for visualization)
f = np.linspace(1e2, 1e5, 100)
w = 2 * np.pi * f
_, H_true = freqs(b, a, worN=w)
_, H_dut, UH_dut = MC_freq(b, a, Uab, use_w=w, return_full_cov=True)
_, H_inv, UH_inv = MC_freq(
    b_inv, a_inv, Uab_inv, dt=1 / fs, use_w=w, return_full_cov=True
)
_, H_reg, UH_reg = MC_freq(
    b_reg, a_reg, Uab_reg, dt=1 / fs, use_w=w, return_full_cov=True
)

UH_dut_diag = np.sqrt(np.diag(UH_dut))
UH_dut_abs = np.abs(real_imag_2_complex(UH_dut_diag))
UH_dut_phase = np.angle(real_imag_2_complex(UH_dut_diag))

UH_inv_diag = np.sqrt(np.diag(UH_inv))
UH_inv_abs = np.abs(real_imag_2_complex(UH_inv_diag))
UH_inv_phase = np.angle(real_imag_2_complex(UH_inv_diag))

UH_reg_diag = np.sqrt(np.diag(UH_reg))
UH_reg_abs = np.abs(real_imag_2_complex(UH_reg_diag))
UH_reg_phase = np.angle(real_imag_2_complex(UH_reg_diag))

# visualize and compare fitted inverse transfer behavior in the frequency domain
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 6), sharex=True)
ax[0].plot(f, np.abs(H_true), "k", label="true DUT", zorder=100)
ax[0].plot(f, np.abs(H_dut), "b", label="fitted DUT", zorder=98)
ax[0].fill_between(
    f,
    np.abs(H_dut) - UH_dut_abs,
    np.abs(H_dut) + UH_dut_abs,
    color="b",
    alpha=0.2,
    zorder=98,
)
ax[0].plot(f, np.abs(H_inv), "r", label="inverse DUT", zorder=99)
ax[0].fill_between(
    f,
    np.abs(H_inv) - UH_inv_abs,
    np.abs(H_inv) + UH_inv_abs,
    color="r",
    alpha=0.2,
    zorder=99,
)
ax[0].plot(f, np.abs(H_reg), "c", label="regularized DUT", zorder=97)
ax[0].fill_between(
    f,
    np.abs(H_reg) - UH_reg_abs,
    np.abs(H_reg) + UH_reg_abs,
    color="c",
    alpha=0.2,
    zorder=97,
)

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_ylabel("frequency response (abs)")
ax[0].legend()

ax[1].plot(f, np.angle(H_true), "k", label="true DUT", zorder=100)
ax[1].plot(f, np.angle(H_dut), "b", label="fitted DUT", zorder=98)
ax[1].fill_between(
    f,
    np.angle(H_dut) - UH_dut_phase,
    np.angle(H_dut) + UH_dut_phase,
    color="b",
    alpha=0.2,
    zorder=98,
)
ax[1].plot(f, np.angle(H_inv), "r", label="inverse DUT", zorder=99)
ax[1].fill_between(
    f,
    np.angle(H_inv) - UH_inv_phase,
    np.angle(H_inv) + UH_inv_phase,
    color="r",
    alpha=0.2,
    zorder=99,
)
ax[1].plot(f, np.angle(H_reg), "c", label="regularized DUT", zorder=97)
ax[1].fill_between(
    f,
    np.angle(H_reg) - UH_reg_phase,
    np.angle(H_reg) + UH_reg_phase,
    color="c",
    alpha=0.2,
    zorder=97,
)

ax[1].set_xscale("log")
ax[1].set_yscale("linear")
ax[1].set_xlabel("frequency (Hz)")
ax[1].set_ylabel("frequency response (phase)")

# generate (*@ \cref{fig:dynunc_inverse_estimation} @*)
if save_figures: 
    fig.savefig("inverse_estimation.pdf", bbox_inches="tight")
else:
    plt.show()


# visualize and compare (only the) uncertainty of fitted inverse transfer behavior in the frequency domain
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 6), sharex=True)
ax[0].plot(f, np.zeros_like(f), "k", label="true DUT", zorder=100)
ax[0].plot(f, UH_dut_abs, "b", label="fitted DUT", zorder=98)
ax[0].plot(f, UH_inv_abs, "r", label="inverse DUT", zorder=99)
ax[0].plot(f, UH_reg_abs, "c", label="regularized DUT", zorder=97)

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_ylabel("unc. of frequency response (abs)")
ax[0].legend()

ax[1].plot(f, np.zeros_like(f), "k", label="true DUT", zorder=100)
ax[1].plot(f, UH_dut_phase, "b", label="fitted DUT", zorder=98)
ax[1].plot(f, UH_inv_phase, "r", label="inverse DUT", zorder=99)
ax[1].plot(f, UH_reg_phase, "c", label="regularized DUT", zorder=97)

ax[1].set_xscale("log")
ax[1].set_yscale("linear")
ax[1].set_xlabel("frequency (Hz)")
ax[1].set_ylabel("unc. of frequency response (phase)")

# generate (*@ \cref{fig:dynunc_inverse_estimation} @*)
if save_figures: 
    fig.savefig("inverse_estimation_unc_only.pdf", bbox_inches="tight")
else:
    plt.show()



# visualize the input estimation
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 6))
ax[0].plot(t, acc_true, "k", label="measurand", zorder=100)
ax[0].plot(t, acc_dut, "b", label="DUT indicated", zorder=98)
ax[0].plot(t, acc_comp, "r", label="DUT compensated", zorder=99)

# zoom
zoomed_indices = zi = np.logical_and(t >= 0.005500, t <= 0.005800)
zoom_highlight = Rectangle(
    (min(t[zi]), -5),
    max(t[zi]) - min(t[zi]),
    10,
    linestyle="--",
    linewidth=1,
    color="r",
    fill=False,
    label="zoom area",
    zorder=101,
)
ax[0].add_patch(zoom_highlight)
ax[0].set_ylabel("acceleration signal")
ax[0].legend()

ax[1].plot(t[zi], acc_true[zi], "k", label="measurand", zorder=100)
ax[1].plot(t[zi], acc_dut[zi], "b", label="DUT indicated", zorder=96)
ax[1].fill_between(
    t[zi],
    acc_dut[zi] - Uacc_dut[zi],
    acc_dut[zi] + Uacc_dut[zi],
    color="b",
    alpha=0.2,
    zorder=95,
)
ax[1].plot(t[zi], acc_comp[zi], "r", label="DUT compensated", zorder=98)
ax[1].fill_between(
    t[zi],
    acc_comp[zi] - Uacc_comp[zi],
    acc_comp[zi] + Uacc_comp[zi],
    color="r",
    alpha=0.2,
    zorder=97,
)
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("acceleration signal")

# generate (*@ \cref{fig:dynunc_input_estimation} @*)
if save_figures: 
    fig.savefig("input_estimation.pdf", bbox_inches="tight")
else:
    plt.show()


# visualize the input estimation
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 6))
#ax[0].plot(t, np.zeros_like(t), "k", label="measurand", zorder=100)
ax[0].plot(t, Uacc_dut, "b", label="DUT indicated", zorder=98)
ax[0].plot(t, Uacc_comp, "r", label="DUT compensated", zorder=99)

# zoom
zoomed_indices = zi = np.logical_and(t >= 0.005500, t <= 0.005800)
zoom_highlight = Rectangle(
    (min(t[zi]), 0.19),
    max(t[zi]) - min(t[zi]),
    0.07,
    linestyle="--",
    linewidth=1,
    color="r",
    fill=False,
    label="zoom area",
    zorder=101,
)
ax[0].add_patch(zoom_highlight)
ax[0].set_ylabel("unc. of acceleration signal")
ax[0].legend()

#ax[1].plot(t[zi], np.zeros_like(t[zi]), "k", label="measurand", zorder=100)
ax[1].plot(t[zi], Uacc_dut[zi], "b", label="DUT indicated", zorder=96)
ax[1].plot(t[zi], Uacc_comp[zi], "r", label="DUT compensated", zorder=98)

ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("unc. of acceleration signal")

# generate (*@ \cref{fig:dynunc_input_estimation} @*)
if save_figures: 
    fig.savefig("input_estimation_unc_only.pdf", bbox_inches="tight")
else:
    plt.show()

