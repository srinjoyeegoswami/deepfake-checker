import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt


def pos_signal(rgb, fps, win_sec=1.6):
  eps = 1e-8
  N = len(rgb)
  L = int(win_sec * fps)
  H = np.zeros(N)
  proj = np.array([[0, 1, -1], [-2, 1, 1]])
  for m in range(0, N - L):
    C = rgb[m : m + L].T
    Cn = C / (C.mean(axis=1, keepdims=True) + eps)
    S = proj @ Cn
    h = S[0] + (S[0].std() / (S[1].std() + eps)) * S[1]
    H[m : m + L] += h - h.mean()
  return H


# Load Subject 3 arrays
rgb = np.load("subject3_rgb.npy")
fps = float(np.load("subject3_fps.npy")[0])
true_bpm = 94.7

# Filter setup
sos = butter(3, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")

# 1. Green Channel Spectrum
g = detrend(rgb[:, 1])
filt_g = sosfiltfilt(sos, g)
spec_g = np.abs(np.fft.rfft(filt_g))
freqs = np.fft.rfftfreq(len(filt_g), d=1 / fps)
band = (freqs >= 0.7) & (freqs <= 3.0)

# 2. POS Spectrum
p = detrend(pos_signal(rgb, fps))
filt_p = sosfiltfilt(sos, p)
spec_p = np.abs(np.fft.rfft(filt_p))

# 3. Side-by-Side Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Left: Green
ax[0].plot(freqs[band] * 60, spec_g[band], color="#1f77b4")
ax[0].axvline(77.1, color="r", linestyle="--", label="Est: 77.1 BPM")
ax[0].axvline(true_bpm, color="g", linestyle="--", label="True: 94.7 BPM")
ax[0].set_title("Raw Green (Motion Corrupted)")
ax[0].set_xlabel("Beats per Minute (BPM)")
ax[0].set_ylabel("Spectral Magnitude")
ax[0].legend()

# Right: POS
ax[1].plot(freqs[band] * 60, spec_p[band], color="#ff7f0e")
ax[1].axvline(92.0, color="r", linestyle="--", label="Est: 92.0 BPM")
ax[1].axvline(true_bpm, color="g", linestyle="--", label="True: 94.7 BPM")
ax[1].set_title("POS Projection (Pulse Recovered)")
ax[1].set_xlabel("Beats per Minute (BPM)")
ax[1].legend()

plt.tight_layout()
plt.savefig("sub3_green_vs_pos.png", dpi=300)
print("Saved sub3_green_vs_pos.png successfully.")