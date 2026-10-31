import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, sosfiltfilt

# --- 1. Define POS function ---
def pos_signal(rgb, fps, win_sec=1.6):
    """POS: combine R, G, B so that motion and lighting cancel out and the pulse remains."""
    eps = 1e-8
    N = len(rgb)
    L = int(win_sec * fps)                  # window length in frames (~1.6 seconds)
    H = np.zeros(N)                         # the output pulse signal
    proj = np.array([[0, 1, -1],            # S1 = G - B
                     [-2, 1, 1]])           # S2 = -2R + G + B
    for m in range(0, N - L):
        C = rgb[m:m + L].T                  # 3 x L: R, G, B over this window
        Cn = C / (C.mean(axis=1, keepdims=True) + eps)   # divide each channel by its mean
        S = proj @ Cn                       # project color channels
        h = S[0] + (S[0].std() / (S[1].std() + eps)) * S[1]   # motion cancellation weight
        H[m:m + L] += h - h.mean()          # overlap-add into output
    return H

# --- CONFIGURATION (Change only these two lines per test) ---
SUBJECT = 3         # Options: 1, 2, or 3
METHOD = "pos"      # Options: "pos" or "green"

# Known ground-truth mean heart rates from gtdump.xmp
GT_MAP = {
    1: 77.3,
    2: 82.6,
    3: 94.7
}
TRUE_BPM = GT_MAP[SUBJECT]

# --- 2. Load cached signals dynamically ---
rgb = np.load(f"subject{SUBJECT}_rgb.npy")
fps = float(np.load(f"subject{SUBJECT}_fps.npy")[0])

if METHOD == "pos":
    signal = pos_signal(rgb, fps)
else:
    signal = rgb[:, 1]                      # raw green channel

print(f"--- Subject: {SUBJECT} | Method: {METHOD} ---")
print(f"frames: {len(signal)} | fps: {fps:.2f} | seconds: {len(signal) / fps:.2f}")

# 3. Detrend
g = detrend(signal)

# 4. Band-pass filter (0.7 to 3.0 Hz, 42 to 180 bpm)
sos = butter(3, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")
filt = sosfiltfilt(sos, g)

# 5. FFT
n = len(filt)
spectrum = np.abs(np.fft.rfft(filt))
freqs = np.fft.rfftfreq(n, d=1 / fps)

# 6. Peak picking
band = (freqs >= 0.7) & (freqs <= 3.0)
peak_hz = freqs[band][np.argmax(spectrum[band])]
est_bpm = peak_hz * 60
error = abs(est_bpm - TRUE_BPM)

print(f"peak frequency: {peak_hz:.3f} Hz  ->  estimated heart rate: {est_bpm:.1f} bpm")
print(f"true heart rate: {TRUE_BPM:.1f} bpm  |  error: {error:.1f} bpm")

# --- 7. Plots ---
t = np.arange(n) / fps
fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(t, signal)
ax[0].set_title(f"1. raw signal ({METHOD})")
ax[1].plot(t, filt)
ax[1].set_title("2. filtered (0.7-3.0 Hz)")
ax[2].plot(freqs[band] * 60, spectrum[band])
ax[2].axvline(est_bpm, color="r", linestyle="--", label=f"estimate {est_bpm:.0f} bpm")
ax[2].axvline(TRUE_BPM, color="g", linestyle="--", label=f"true {TRUE_BPM:.0f} bpm")
ax[2].set_xlabel("beats per minute")
ax[2].set_title("3. spectrum")
ax[2].legend()

plt.tight_layout()
plot_filename = f"bpm_result_sub{SUBJECT}_{METHOD}.png"
plt.savefig(plot_filename)
print(f"saved {plot_filename}")