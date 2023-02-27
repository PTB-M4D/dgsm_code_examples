# -*- coding: utf-8 -*-
"""
    Code to generate figures in Chapter 2 (Fundamentals) related to different representations of an LTI system
"""

import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt

import PyDynamic.misc.SecondOrderSystem as sos
from PyDynamic.misc.testsignals import *

do_save = True

rst = np.random.RandomState(10)

plt.rcParams.update({'font.size':22,
                     'figure.figsize': [16,9],
                     'legend.fontsize': 'large',
                     'lines.linewidth': 2
                     })

# parameters of simulated measurement
Fs = 500e3      # sampling frequency (in Hz)
Ts = 1 / Fs     # sampling interval length (in s)

# sensor/measurement system parameters with associated uncertainties
f0 = 20e3; uf0 = 0.01*f0            # resonance frequency
S0 = 1.0; uS0= 0.001*S0           # static gain
delta = 0.1; udelta = 0.1*delta  # damping

f = np.linspace(0, 120e3, 500)      # frequencies at which to calculate frequency response
Hf = sos.sos_FreqResp(S0, delta, f0, f)

fig, ax1 = plt.subplots()
ax1.semilogy(f, np.abs(Hf), color="b")
ax1.set_xlabel("frequency [Hz]")
ax1.set_ylabel("magnitude of frequency response [a.u.]", color="b")
ax2 = ax1.twinx()
ax2.plot(f, np.angle(Hf), color="k")
ax2.set_ylabel("phase of frequency response [rad]")
if do_save:
    plt.savefig("magn_phase_response.pdf", bbox_inches="tight")

# transform continuous system to digital filter
bc, ac = sos.sos_phys2filter(S0,delta,f0)   # calculate analogue filter coefficients
b, a = dsp.bilinear(bc, ac, Fs)             # transform to digital filter coefficients

# simulate input and output signals
time = np.arange(0, 0.8e-3 - Ts, Ts)                              # time values
x = shocklikeGaussian(time, t0 = 0.2e-3, sigma = 1.5e-5, m0=0.8)    # input signal
y = dsp.lfilter(b, a, x)                                        # output signal

fig, ax1 = plt.subplots()
ax1.plot(time, x, color="g", label="input signal [m/s^2]")
ax1.plot(time, y, color="r", label="output signal [a.u.]")
ax1.set_ylabel("signal amplitude")
ax1.set_xlabel("time [s]")
ax1.legend()
if do_save:
    plt.savefig("accelerometer_example.pdf", bbox_inches="tight")


# step response
sx = rect(time, 0.01e-3, 3e-3, 1)
sy = dsp.lfilter(b, a, sx)

fig, ax1 = plt.subplots()
ax1.plot(time, sx, color="g", label="step input")
ax1.plot(time, sy, color="r", label="step response")
ax1.set_ylabel("signal amplitude")
ax1.set_xlabel("time [s]")
ax1.legend()
if do_save:
    plt.savefig("stepresponse.pdf", bbox_inches="tight")

# impulse response
_, iy = dsp.impulse((bc, ac), T=time)
ix = np.zeros_like(time)
ix[0] = np.abs(np.max(iy))
iy = np.roll(iy,1)

fig, ax1 = plt.subplots()
ax1.plot(time, ix, color="g", label="impulse input")
ax1.plot(time, iy, color="r", label="impulse response")
ax1.set_ylabel("signal amplitude")
ax1.set_xlabel("time [s]")
ax1.legend()
if do_save:
    plt.savefig("impulseresponse.pdf", bbox_inches="tight")

plt.show()