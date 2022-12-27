import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import PyDynamic.misc.SecondOrderSystem as sos
from scipy.signal import cheby2, freqz

###############################
### calculation part ##########
###############################

# parameters of simulated measurement
fs = 500e3      # sampling frequency (in Hz)
Ts = 1 / fs     # sampling interval length (in s)

# sensor/measurement system parameters with associated uncertainties
f0 = 20e3;           # resonance frequency
S0 = 1.0;            # static gain
delta = 0.1;         # damping

f = np.linspace(0, 120e3, 500)      # frequencies at which to calculate frequency response
w = 2 * np.pi * f
H = sos.sos_FreqResp(S0, delta, f0, f)
H_inv = 1/H

# lowpass regularization
b_low, a_low = cheby2(4, 30, 60e3, btype="low", analog=False, output="ba", fs=fs)
_, Hlow = freqz(b_low, a_low, worN=w, fs=2*np.pi*fs)
H_reg = H_inv * Hlow * H

###############################
### visualization part ########
###############################
save_figures = True

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
ax.plot(f, np.abs(H), "k", label="system frequency response", zorder=100)
ax.plot(f, np.abs(H_inv), "r", label="inverse frequency response", zorder=99)
ax.plot(f, np.abs(H_reg), "c", label="regularized compensation", zorder=97)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("magnitude of frequency response")
ax.set_xlabel("frequency [Hz]")
ax.legend()

if save_figures: 
    fig.savefig("compensation_example.pdf", bbox_inches="tight")
else:
    plt.show()
