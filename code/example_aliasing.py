import numpy as np
import matplotlib.pyplot as plt
from PyDynamic.misc.impinvar import impinvar
import PyDynamic.misc.SecondOrderSystem as sos
import scipy.signal as dsp

save_figures = True

# Continuous-time transfer function
f0 = 2e3;           # resonance frequency
S0 = 1.0;            # static gain
delta = 0.1;         # damping
bc, ac = sos.sos_phys2filter(S0, delta, f0)

# Frequency range for the plots
fs = 10 * f0
Ts = 1/fs
f = np.linspace(0, fs/2, 1000)
w = 2*np.pi*f
_, Hc = dsp.freqs(bc, ac, worN=w)

# Apply method of impulse invariance to calculate digital filter coefficients
bd, ad = impinvar(bc, ac, fs)
_, Hd = dsp.freqz(bd, ad, worN=w, fs=2*np.pi*fs)

fs2 = 5*f0
bd2, ad2 = impinvar(bc, ac, fs2)
_, Hd2 = dsp.freqz(bd2, ad2, worN=w, fs=2*np.pi*fs2)

# Plot for continuous-time transfer function
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(f*1e-3, np.abs(Hc), label="continuous-time")
ax.plot(f*1e-3, np.abs(Hd), label=f"discrete-time (Ts = {Ts})")
ax.plot(f*1e-3, np.abs(Hd2), label=f"discrete-time (Ts = {1/fs2})")
ax.set_xlabel("frequency in kHz")
ax.set_ylabel("frequency response magnitude")
ax.legend()

if save_figures: 
    fig.savefig("aliasing_example.pdf", bbox_inches="tight")
else:
    plt.show()
