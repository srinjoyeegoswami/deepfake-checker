import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from pipeline import pos_signal

TARGETS = {
    "subject3": 94.7,
    "subject4": 67.0,
    "subject7": 72.2,
}

fig, axes = plt.subplots(len(TARGETS), 1, figsize=(11, 9), sharex=True)

for ax, (sub, truth) in zip(axes, TARGETS.items()):
    data = np.load(f"cache/{sub}.npz")
    rgb, fps = data["rgb"], float(data["fps"])
    
    # 1. Bandpass setup (0.7 - 3.0 Hz -> 42 - 180 BPM)
    sos = sp.butter(3, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")
    
    # 2. Green signal spectrum
    green_detrended = sp.detrend(rgb[:, 1])
    green_filtered = sp.sosfiltfilt(sos, green_detrended)
    spec_green = np.abs(np.fft.rfft(green_filtered))
    
    # 3. POS signal spectrum
    pos_raw = pos_signal(rgb, fps)
    pos_detrended = sp.detrend(pos_raw)
    pos_filtered = sp.sosfiltfilt(sos, pos_detrended)
    spec_pos = np.abs(np.fft.rfft(pos_filtered))
    
    # 4. Frequency axis in BPM
    freqs = np.fft.rfftfreq(len(green_filtered), d=1.0 / fps) * 60.0
    band = (freqs >= 42.0) & (freqs <= 180.0)
    
    f_band = freqs[band]
    norm_green = spec_green[band] / np.max(spec_green[band])
    norm_pos = spec_pos[band] / np.max(spec_pos[band])
    
    peak_green = f_band[np.argmax(norm_green)]
    peak_pos = f_band[np.argmax(norm_pos)]
    
    # 5. Plotting
    ax.plot(f_band, norm_green, label=f"Green Peak: {peak_green:.1f} BPM", color="#2ca02c", linewidth=1.7)
    ax.plot(f_band, norm_pos, label=f"POS Peak: {peak_pos:.1f} BPM", color="#1f77b4", linewidth=1.7)
    ax.axvline(truth, color="red", linestyle="--", linewidth=1.5, label=f"Ground Truth: {truth:.1f} BPM")
    
    ax.set_ylabel("Norm. Magnitude")
    ax.set_title(f"{sub.capitalize()} Spectral Power (Truth: {truth:.1f} BPM)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(-0.05, 1.15)

axes[-1].set_xlabel("Heart Rate (BPM)", fontsize=11)
plt.xlim(42, 180)
plt.tight_layout()
plt.savefig("spectra_comparison.png", dpi=200)
print("Spectra successfully generated and saved to spectra_comparison.png")