import numpy as np
import os
import pandas as pd
from pipeline import extract_rgb_crop, fill_gaps, pos_signal
from scipy.signal import butter, detrend, sosfiltfilt

os.makedirs("cache", exist_ok=True)

meta = pd.read_csv("dfdc_meta.csv")
sample = pd.concat([
    meta[meta.label == "REAL"].head(20),
    meta[meta.label == "FAKE"].head(20),
])

def peak_ratio(rgb, fps):
    sig = detrend(pos_signal(rgb, fps))
    sos = butter(3, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")
    sig = sosfiltfilt(sos, sig)
    spec = np.abs(np.fft.rfft(sig)) ** 2
    f = np.fft.rfftfreq(len(sig), d=1 / fps)
    band = (f >= 0.7) & (f <= 3.0)
    return (
        spec[band].max() / spec[band].sum(),
        f[band][spec[band].argmax()] * 60,
    )

rows = []
print(f"Rerunning crop-based sanity check on {len(sample)} clips...")

for idx, (_, r) in enumerate(sample.iterrows()):
    path = f"data/dfdc/train_sample_videos/{r.filename}"
    cache_file = f"cache/dfdc_crop_{r.filename}.npz"
    try:
        rgb, fps, missing = extract_rgb_crop(path, cache_file)
        rgb = fill_gaps(rgb)
        
        if rgb is None:
            # Dropped because missing fraction exceeded threshold
            rows.append((r.filename, r.label, fps, np.nan, np.nan, np.nan, missing))
            print(f"[{idx+1}/{len(sample)}] {r.filename} ({r.label}): DROPPED (missing={missing:.1%})")
            continue
            
        ratio, bpm = peak_ratio(rgb, fps)
        rows.append((r.filename, r.label, fps, len(rgb), ratio, bpm, missing))
        print(f"[{idx+1}/{len(sample)}] {r.filename} ({r.label}): fps={fps:.1f} ratio={ratio:.3f} bpm={bpm:.1f} missing={missing:.1%}")
    except Exception as e:
        print(f"[{idx+1}/{len(sample)}] Error on {r.filename}: {e}")

df = pd.DataFrame(rows, columns=["file", "label", "fps", "frames", "ratio", "bpm", "missing"])
df.to_csv("sanity_check_crop.csv", index=False)

print("\n=== Drop Rate by Class (NaN in ratio) ===")
print(df.groupby("label")["ratio"].apply(lambda s: s.isna().sum()))

print("\n=== Summary by Class (Valid Clips Only) ===")
print(df.dropna(subset=["ratio"]).groupby("label")[["ratio", "bpm", "missing"]].describe().T)