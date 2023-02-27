import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PyDynamic.misc.tools import real_imag_2_complex
from PyDynamic.model_estimation import LSIIR
from PyDynamic.uncertainty.propagate_DFT import GUM_DFT, DFT_deconv, GUM_DFTfreq
from scipy.signal import chirp, freqs, lsim

from helper_methods import MC_freq

###############################
### calculation part ##########
###############################

# true device under test (DUT) transfer behavior
delta_true = 8.3e-3  # 1
f0_true = 29.4e3  # Hz
omega0_true = 2 * np.pi * f0_true  # Hz
S0_true = 0.985  # 1

a = np.array([1.0, 2 * delta_true * omega0_true, omega0_true ** 2])
b = np.array([S0_true * omega0_true ** 2])

# true acceleration
fs = 500e3  # Hz
t = np.arange(0, 0.01, 1 / fs)
acc_true = chirp(t, f0=10, t1=t[-1], f1=40e3, method="linear", phi=-90)

# measured reference signal
u_ref = 0.05
acc_ref = acc_true + u_ref * np.random.randn(acc_true.size)

# measured DUT signal
t, acc_dut, xout = lsim((b, a), U=acc_true, T=t)

# transfrom to frequency domain
f = GUM_DFTfreq(len(t), dt=1 / fs)
F_ref, UF_ref = GUM_DFT(acc_ref, u_ref ** 2)
F_dut, UF_dut = GUM_DFT(acc_dut, 0.0)
H, UH = DFT_deconv(F_ref, F_dut, UF_ref, UF_dut)
H = real_imag_2_complex(H)


# calibration (fit IIR to empirical spectrum)
# only fit up to 40kHz, as noise is expected above
fi = fit_indices = f < 40e3
fifi = np.hstack((fi, fi))
b_discrete, a_discrete, tau, Uab_discrete = LSIIR(
    H[fi],
    UH=UH[fifi, :][:, fifi],
    f=f[fi],
    Nb=1,
    Na=2,
    Fs=fs,
    mc_runs=500,
    verbose=False,
)

# calculate true transfer functions of true sensor and DUT (only for visualization)
W, H_true = freqs(b, a, worN=2 * np.pi * f)
W, H_dut, UH_dut = MC_freq(
    b_discrete,
    a_discrete,
    Uab_discrete,
    dt=1 / fs,
    use_w=2 * np.pi * f,
    return_full_cov=True,
)
UH_dut_diag = np.sqrt(np.diag(UH_dut))
UH_dut_abs = np.abs(real_imag_2_complex(UH_dut_diag))
UH_dut_phase = np.angle(real_imag_2_complex(UH_dut_diag))

###############################
### visualization part ########
###############################
save_figures = True

# print numerical values
np.set_printoptions(precision=2)

print(f"{b_discrete=}")
print(f"{a_discrete=}")
print(f"{Uab_discrete=}\n")

# visualize time series of signals used for calibration
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))
ax[0].plot(t, acc_true, "k", label="measurand", zorder=100)
ax[0].plot(t, acc_ref, "r", label="reference", zorder=99)
ax[0].plot(t, acc_dut, "b", label="device under test")

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
ax[0].set_ylabel("acceleration signal (a.u.)")
ax[0].legend(loc="upper left")

ax[1].plot(t[zi], acc_true[zi], "k", label="measurand", zorder=100)
ax[1].plot(t[zi], acc_ref[zi], "r", label="reference", zorder=99)
ax[1].fill_between(
    t[zi], acc_ref[zi] - u_ref, acc_ref[zi] + u_ref, color="r", alpha=0.2, zorder=98
)
ax[1].plot(t[zi], acc_dut[zi], "b", label="device under test")
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("acceleration signal (a.u.)")

# generate (*@ \cref{fig:dynunc_calibration_input} @*)
if save_figures: 
    fig.savefig("calibration_input.pdf", bbox_inches="tight")
else:
    plt.show()

# visualize and compare fitted transfer behavior in the frequency domain
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 5), sharex=True)
ax[0].plot(f, np.abs(H_true), "k", label="true transfer function of DUT", zorder=100)
ax[0].plot(f, np.abs(H), "r", label="experimental data", zorder=98)
ax[0].plot(f, np.abs(H_dut), "b", label="fitted transfer function", zorder=99)
ax[0].fill_between(
    f,
    np.abs(H_dut) - UH_dut_abs,
    np.abs(H_dut) + UH_dut_abs,
    color="b",
    alpha=0.2,
    zorder=99,
)

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_ylabel("frequency response (abs) (a.u.)")
ax[0].legend()

ax[1].plot(f, np.angle(H_true), "k", label="true transfer function of DUT", zorder=100)
ax[1].plot(f, np.angle(H), "r", label="experimental data", zorder=98)
ax[1].plot(f, np.angle(H_dut), "b", label="fitted transfer function", zorder=99)
ax[1].fill_between(
    f,
    np.angle(H_dut) - UH_dut_phase,
    np.angle(H_dut) + UH_dut_phase,
    color="b",
    alpha=0.2,
    zorder=99,
)

ax[1].set_xscale("log")
ax[1].set_yscale("linear")
ax[1].set_xlabel("frequency (Hz)")
ax[1].set_ylabel("frequency response (phase) (rad)")

# generate (*@ \cref{fig:dynunc_calibration_result} @*)
if save_figures: 
    fig.savefig("calibration_result.pdf", bbox_inches="tight")
else:
    plt.show()
