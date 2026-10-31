import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents GUI / display backend crashes when saving plots
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, sosfiltfilt
from pipeline import extract_rgb, pos_signal

VIDEO = "self/self_exercise.mp4"      # change to your file
TRUE_BPM = 143.0

rgb, fps = extract_rgb(VIDEO, "cache/self_exercise.npz")

def clean(sig, fps):
    sig = detrend(sig)
    sos = butter(3, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")
    return sosfiltfilt(sos, sig)

def spectrum(sig, fps):
    f = np.fft.rfftfreq(len(sig), d=1 / fps) * 60          # in bpm
    s = np.abs(np.fft.rfft(sig))
    band = (f >= 42) & (f <= 180)
    return f[band], s[band]

pos = clean(pos_signal(rgb, fps), fps)
green = clean(rgb[:, 1], fps)

fig, ax = plt.subplots(3, 1, figsize=(10, 11))

# 1. Whole-clip spectra for both methods
for sig, name in [(green, "green"), (pos, "POS")]:
    f, s = spectrum(sig, fps)
    ax[0].plot(f, s / s.max(), label=name)
ax[0].axvline(TRUE_BPM, color="g", ls="--", label=f"true {TRUE_BPM:.0f}")
ax[0].axvline(TRUE_BPM / 2, color="gray", ls=":", label="half of true")
ax[0].set_title("Whole-clip spectrum: is there any peak at the true rate?")
ax[0].set_xlabel("bpm"); ax[0].legend()

# 2. Windowed estimate: does the heart rate drift during the clip?
win, hop = int(10 * fps), int(2 * fps)
centres, ests = [], []
for start in range(0, len(pos) - win, hop):
    f, s = spectrum(pos[start:start + win], fps)
    ests.append(f[np.argmax(s)])
    centres.append((start + win / 2) / fps)
ax[1].plot(centres, ests, "o-", label="POS, 10 s windows")
ax[1].axhline(TRUE_BPM, color="g", ls="--", label="true (single value)")
ax[1].axhline(TRUE_BPM / 2, color="gray", ls=":")
ax[1].set_title("Windowed estimate over time (each step is ~6 bpm resolution)")
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("bpm"); ax[1].legend()

# 3. Filtered signal
ax[2].plot(np.arange(len(pos)) / fps, pos)
ax[2].set_title("Filtered POS signal: a regular wave, or irregular bursts?")
ax[2].set_xlabel("time (s)")

plt.tight_layout()
plt.savefig("diagnose_post_exercise.png")
print("saved diagnose_post_exercise.png")