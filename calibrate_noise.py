import numpy as np
import os
from scipy.signal import detrend, butter, sosfiltfilt
from pipeline import pos_signal

def peak_ratio(rgb, fps):
    sig = detrend(pos_signal(rgb, fps))
    sos = butter(3, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")
    sig = sosfiltfilt(sos, sig)
    spec = np.abs(np.fft.rfft(sig)) ** 2
    f = np.fft.rfftfreq(len(sig), d=1 / fps)
    band = (f >= 0.7) & (f <= 3.0)
    return spec[band].max() / spec[band].sum()

fps = 29.97

# 1. Null distribution: white noise passed through the identical filtering chain
rng = np.random.default_rng(0)
null = [peak_ratio(150 + rng.normal(0, 1, (300, 3)), fps) for _ in range(500)]
print("NOISE   ratio: mean %.3f | 5th-95th pct %.3f-%.3f" %
      (np.mean(null), *np.percentile(null, [5, 95])))

# 2. Ground-truth benchmark: 10 s windows from your cached UBFC subjects
for name in ["subject1", "subject2", "subject3"]:
    cache_path = f"cache/{name}.npz"
    if not os.path.exists(cache_path):
        print(f"{name}: cache file not found at {cache_path}, skipping.")
        continue
    d = np.load(cache_path)
    rgb, f_ = d["rgb"], float(d["fps"])
    vals = [peak_ratio(rgb[s:s + 300], f_) for s in range(0, len(rgb) - 300, 150)]
    print(f"{name}: mean ratio {np.mean(vals):.3f} over {len(vals)} windows")