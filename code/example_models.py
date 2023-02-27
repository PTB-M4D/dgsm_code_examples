import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg.misc import norm

from helper_methods import MC_cont2discrete, MC_freq, MC_impulse

###############################
### calculation part ##########
###############################

# estimated parameter values
delta_hat = 0.01  # 1
f0_hat = 30e3  # Hz
omega0_hat = 2 * np.pi * f0_hat  # Hz
S0_hat = 1  # 1
U_theta_hat = np.diag(np.square([0.1 * delta_hat, 0.03 * f0_hat, 0.01 * S0_hat]))

# estimated filter coefficients
a = np.array([1.0, 2 * delta_hat * omega0_hat, omega0_hat ** 2])
b = np.array([S0_hat * omega0_hat ** 2])
C = np.array(
    [
        [2 * omega0_hat, 2 * delta_hat, 0],
        [0, 2 * omega0_hat, 0],
        [0, 2 * omega0_hat * S0_hat, omega0_hat ** 2],
    ]
)  # sensitivities
ab = np.hstack([a[1:], b])
Uab = C @ U_theta_hat @ C.T

# discrete filter coefficients
fs = 500e3  # Hz
b_discrete, a_discrete, Uab_discrete = MC_cont2discrete(
    b, a, Uab, dt=1.0 / fs, runs=500
)

# get spectra
W_conti, H_conti, UH_conti_diag = MC_freq(b, a, Uab)
W_discrete, H_discrete, UH_discrete_diag = MC_freq(
    b_discrete, a_discrete, Uab_discrete, dt=1.0 / fs, use_w=W_conti
)

# get impulse / step response
t_discrete = np.arange(0, 1000 / fs, 1 / fs)
t_discrete, h_discrete, uh_discrete_diag = MC_impulse(
    b_discrete, a_discrete, Uab_discrete, dt=1.0 / fs, use_t=t_discrete
)

###############################
### visualization part ########
###############################
save_figures = False

# print numerical values
np.set_printoptions(precision=2)

print(f"{b=}")
print(f"{a=}")
print(f"{Uab=}\n")

print(f"{b_discrete=}")
print(f"{a_discrete=}")
print(f"{Uab_discrete=}")

# visualize frequency spectrum of analog and discrete filters
k = 1  # 2..4 for better visualization
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 5))
ax[0].plot(W_conti / (2 * np.pi), np.abs(H_conti), "k", label="continuous IIR")
ax[0].fill_between(
    W_conti / (2 * np.pi),
    np.abs(H_conti) - k * UH_conti_diag,
    np.abs(H_conti) + k * UH_conti_diag,
    color="k",
    alpha=0.2,
)
ax[0].plot(W_discrete / (2 * np.pi), np.abs(H_discrete), "r", label="discrete IIR")
ax[0].fill_between(
    W_discrete / (2 * np.pi),
    np.abs(H_discrete) - k * UH_discrete_diag,
    np.abs(H_discrete) + k * UH_discrete_diag,
    color="r",
    alpha=0.2,
)

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_ylabel("frequency response (a.u.)")
ax[0].legend()

ax[1].plot(W_conti / (2 * np.pi), UH_conti_diag, color="k")
ax[1].plot(W_discrete / (2 * np.pi), UH_discrete_diag, "or", markersize=4)
ax[1].set_xlabel("frequency (Hz)")
ax[1].set_ylabel("uncertainty (a.u.)")

# generate (*@ \cref{fig:dynunc_frequency_response} @*)
if save_figures: 
    fig.savefig("frequency_response.pdf", bbox_inches="tight")
else:
    plt.show()


# visualize impulse response
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 5))

ax[0].plot(t_discrete, h_discrete, "k", label="continuous")
ax[0].fill_between(
    t_discrete,
    h_discrete - k * uh_discrete_diag,
    h_discrete + k * uh_discrete_diag,
    color="k",
    alpha=0.2,
)

ax[0].set_ylabel("step response (a.u.)")
ax[0].legend()


ax[1].plot(t_discrete, uh_discrete_diag, color="k")
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("uncertainty (a.u.)")

# generate (*@ \cref{fig:dynunc_impulse_response} @*)
if save_figures: 
    fig.savefig("impulse_response.pdf", bbox_inches="tight")
else:
    plt.show()
