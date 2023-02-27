import matplotlib.pyplot as plt
import numpy as np
from PyDynamic.uncertainty.propagate_filter import IIR_get_initial_state, IIRuncFilter
from scipy.signal import lfilter, lfilter_zi, zpk2tf, cont2discrete

from helper_methods import MC_inverse_iir

# measurand signal
N = 200
dt = 0.05
t = dt * np.arange(N)
x = np.full(N, 1.0)
x[N//3:2*N//3] = 0.0
ux = np.full_like(x, 0.002)

# sensor transfer behavior (second order lowpass with resonance)
wc = 8  # / 2pi
zeros = np.array([3.5]) * wc
poles = np.array([-0.25+1j, -0.25-1j]) * wc
b_gain, a_gain = zpk2tf(zeros, poles, 1.0)
gain = a_gain[-1] / b_gain[-1]   # ensure stationary accuracy

b, a, _ = cont2discrete(zpk2tf(zeros, poles, gain), dt=dt)
b = np.squeeze(b)
Uba = np.diag(np.hstack((np.full(len(b), 1e-8), np.full(len(a)-1, 1e-8))))

# sensor indication signal
state0 = x[0] * lfilter_zi(b, a)
y, _ = lfilter(b, a, x, zi=state0)
uy = np.full_like(y, 5 / 2**8)  # from 8-bit quantization

# get inverse filter
b_inv, a_inv, uab_inv = MC_inverse_iir(b, a, Uba, dt_input=dt, dt_IIR=dt, tau=1)

# obtain estimate of measurand by applying inverse filter
init_state = IIR_get_initial_state(b_inv, a_inv, Uab=uab_inv, x0=y[0], U0=uy[0])
x_hat, ux_hat, _ = IIRuncFilter(y, uy, b_inv, a_inv, Uab=uab_inv, kind="diag", state=init_state)

# utility function
def plot_unc(ax, x, y, uy, color="r", label=""):
    ax[0].plot(x, y, color=color, label=label)
    ax[0].fill_between(x, y-uy, y+uy, color=color, alpha=0.3)
    #ax[1].plot(x, uy, color=color)
    return ax

# visualize
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,5))
ax = plot_unc(ax, t, x, ux, 'k', "measurand")
ax = plot_unc(ax, t, y, uy, 'r', "indication")
ax = plot_unc(ax, t, x_hat, ux_hat, 'purple', "estimate")

ax[1].plot(t, ux_hat, color="purple")

ax[0].legend()
ax[0].set_ylabel("signals (a.u.)")
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("unc. of estimate (a.u.)")

plt.tight_layout()
#plt.show()
plt.savefig("full_example.pdf")
