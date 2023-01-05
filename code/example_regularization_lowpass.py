import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import PyDynamic.misc.SecondOrderSystem as sos
from PyDynamic.model_estimation.fit_filter import LSFIR
from PyDynamic.misc.testsignals import GaussianPulse
from PyDynamic.misc.filterstuff import kaiser_lowpass
from scipy.signal import cheby2, freqz, bilinear, lfilter

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
bc, ac = sos.sos_phys2filter(S0, delta, f0)
b, a = bilinear(bc, ac, fs)

H_inv = 1/H

# FIR filter fitted to reciprocal system
nFIR = 100
tau = 10
b_inv,_ = LSFIR(H, nFIR, f, fs, tau, inv=True)

# lowpass regularization - high frequency cut-off
low_order = 100
fcut = 60e3
blow, lshift = kaiser_lowpass(low_order, fcut, fs)
shift = int(lshift + tau)
_, Hlow = freqz(blow, 1.0, worN=w, fs=2*np.pi*fs)
H_reg = H_inv * Hlow * H

# lowpass regularization - medium frequency cut-off
low_order = 100
fcut2 = 30e3
blow2, lshift2 = kaiser_lowpass(low_order, fcut2, fs)
shift2 = int(lshift2 + tau)
_, Hlow2 = freqz(blow2, 1.0, worN=w, fs=2*np.pi*fs)
H_reg2 = H_inv * Hlow2 * H

# lowpass regularization - low frequency cut-off
low_order = 100
fcut3 = 20e3
blow3, lshift3 = kaiser_lowpass(low_order, fcut3, fs)
shift3 = int(lshift3 + tau)
_, Hlow3 = freqz(blow3, 1.0, worN=w, fs=2*np.pi*fs)
H_reg3 = H_inv * Hlow3 * H

# test signal measurand
N = 200
time = Ts * np.arange(N)
t0 = time[N//4]
m0 = 1.1
width = 10*Ts
x = GaussianPulse(time, t0, m0, width, noise=0.00)

# test signal measurement
y = lfilter(b, a, x)
# test signal estimate of measurand
xc = lfilter(b_inv, 1.0, y)
xhat = np.roll(lfilter(blow, 1.0, xc),-shift)
xhat2= np.roll(lfilter(blow2, 1.0, xc),-shift2)
xhat3= np.roll(lfilter(blow3, 1.0, xc),-shift3)

# estimation error
rms = lambda v,w: np.sqrt(np.mean((v-w)**2))
xerr= rms(x,xhat)
xer2= rms(x,xhat2)
xer3= rms(x,xhat3)

print("rms error of estimation for different values of lowpass cut-off frequency")
print(f"for fcut={fcut*1e-3} kHz: {xerr*1e3}1e-3")
print(f"for fcut={fcut2*1e-3} kHz: {xer2*1e3}1e-3")
print(f"for fcut={fcut3*1e-3} kHz: {xer3*1e3}1e-3")

###############################
### visualization part ########
###############################
save_figures = False

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(time*1e3, x, "b", label="measurand")
ax.plot(time*1e3, y, "g", label="indicated values")

ax.set_xlabel("time in ms")
ax.set_ylabel("signal amplitude in a.u.")
ax.legend()

if save_figures:
    fig.savefig("regularization_example_input-output.pdf", bbox_inches="tight")


fig, ax = plt.subplots(figsize=(9,6))
ax.plot(time*1e3, x, "b", label="measurand")
ax.plot(time*1e3, y, "g", label="indicated values")
ax.plot(time*1e3, xhat, "c", label="estimate of measurand (high cut-off lowpass)")
ax.plot(time*1e3, xhat2, "m", label="estimate of measurand (medium cut-off lowpass)")
ax.plot(time*1e3, xhat3, "y", label="estimate of measurand (low cut-off lowpass)")

ax.set_xlabel("time in ms")
ax.set_ylabel("signal amplitude in a.u.")
ax.legend()

if save_figures:
    fig.savefig("regularization_example_estimates.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(time*1e3, np.abs(x - xhat), "c", label="estimation error (high cut-off lowpass)")
ax.plot(time*1e3, np.abs(x - xhat2), "m", label="estimation error (medium cut-off lowpass)")
ax.plot(time*1e3, np.abs(x - xhat3), "y", label="estimation error (low cut-off lowpass)")

ax.set_xlabel("time in ms")
ax.set_ylabel("estimation error in a.u.")
ax.legend()

if save_figures:
    fig.savefig("regularization_example_esterror.pdf", bbox_inches="tight")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
ax.plot(f, np.abs(H), "b", label="system frequency response", zorder=100)
ax.plot(f, np.abs(H_inv), "r", label="inverse frequency response", zorder=99)
ax.plot(f, np.abs(H_reg), "c", label="regularized compensation (high cut-off lowpass)", zorder=97)
ax.plot(f, np.abs(H_reg2), "m", label="regularized compensation (medium cut-off lowpass)", zorder=97)
ax.plot(f, np.abs(H_reg3), "y", label="regularized compensation (low cut-off lowpass)", zorder=97)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("magnitude of frequency response")
ax.set_xlabel("frequency [Hz]")
ax.legend()

if save_figures: 
    fig.savefig("regularization_example_freqdomain.pdf", bbox_inches="tight")
else:
    plt.show()
