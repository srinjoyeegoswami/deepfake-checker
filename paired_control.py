import os
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, detrend, sosfiltfilt
from pipeline import extract_rgb_crop, fill_gaps, pos_signal

DATA_DIR = "data/dfdc/train_sample_videos/"
meta = pd.read_csv("dfdc_meta.csv")
sample = pd.concat([
    meta[meta.label == "REAL"].head(20),
    meta[meta.label == "FAKE"].head(20),
])

def compute_ratio(rgb, fps):
    sig = detrend(pos_signal(rgb, fps))
    sos = butter(3, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")
    sig = sosfiltfilt(sos, sig)
    spec = np.abs(np.fft.rfft(sig)) ** 2
    f = np.fft.rfftfreq(len(sig), d=1 / fps)
    band = (f >= 0.7) & (f <= 3.0)
    return spec[band].max() / spec[band].sum()

def extract_bg_rgb(vid_path, patch_size=40):
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        patch = frame[10:10 + patch_size, 10:10 + patch_size]
        frames.append(patch.mean(axis=(0, 1)))
    cap.release()
    return np.array(frames)[:, ::-1], fps

records = []
print(f"Running paired analysis across all {len(sample)} clips...")

for idx, (_, r) in enumerate(sample.iterrows()):
    vid_path = os.path.join(DATA_DIR, r.filename)
    cache_path = f"cache/dfdc_crop_{r.filename}.npz"
    try:
        # 1. Forehead with crop-based tracker
        fh_rgb, fps, missing = extract_rgb_crop(vid_path, cache_path)
        fh_rgb = fill_gaps(fh_rgb)
        if fh_rgb is None:
            continue
        fh_ratio = compute_ratio(fh_rgb, fps)
        
        # 2. Top-left background control patch
        bg_rgb, _ = extract_bg_rgb(vid_path)
        # Match lengths if slightly mismatched
        min_len = min(len(fh_rgb), len(bg_rgb))
        bg_ratio = compute_ratio(bg_rgb[:min_len], fps)
        
        diff = fh_ratio - bg_ratio
        records.append({
            "file": r.filename,
            "label": r.label,
            "forehead_ratio": fh_ratio,
            "background_ratio": bg_ratio,
            "diff": diff
        })
        print(f"[{idx+1}/40] {r.filename} | FH: {fh_ratio:.3f} | BG: {bg_ratio:.3f} | Diff: {diff:+.3f}")
    except Exception as e:
        print(f"Error on {r.filename}: {e}")

df = pd.DataFrame(records)
df.to_csv("paired_control_results.csv", index=False)

print("\n=== Paired Analysis (n = %d) ===" % len(df))
print(df[["forehead_ratio", "background_ratio", "diff"]].describe().T)

t_stat, p_val = stats.ttest_rel(df["forehead_ratio"], df["background_ratio"])
print(f"\nPaired t-test: t = {t_stat:.3f}, p-value = {p_val:.4f}")