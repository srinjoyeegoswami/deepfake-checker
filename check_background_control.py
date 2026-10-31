import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt
from pipeline import pos_signal

DATA_DIR = "data/dfdc/train_sample_videos/"
meta = pd.read_csv("dfdc_meta.csv")

# Use the same 40 clips analyzed previously
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

def extract_background_rgb(vid_path, patch_size=40):
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    frames = []
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Take a patch from the top-left background (outside face bounds)
        patch = frame[10:10 + patch_size, 10:10 + patch_size]
        frames.append(patch.mean(axis=(0, 1)))  # Mean BGR
        
    cap.release()
    rgb = np.array(frames)[:, ::-1]  # Convert BGR to RGB
    return rgb, fps

results = []
print(f"Extracting background control on {len(sample)} clips...")

for idx, (_, r) in enumerate(sample.iterrows()):
    vid_file = os.path.join(DATA_DIR, r.filename)
    try:
        bg_rgb, fps = extract_background_rgb(vid_file)
        bg_ratio = compute_ratio(bg_rgb, fps)
        
        # Load cached forehead signal for direct pair comparison
        forehead_cache = f"cache/dfdc_{r.filename}.npz"
        forehead_ratio = np.nan
        if os.path.exists(forehead_cache):
            fh_data = np.load(forehead_cache)
            fh_rgb = fh_data["rgb"]
            if (fh_rgb.sum(axis=1) != 0).mean() > 0.5:
                forehead_ratio = compute_ratio(fh_rgb, fps)
                
        results.append({
            "file": r.filename,
            "label": r.label,
            "forehead_ratio": forehead_ratio,
            "background_ratio": bg_ratio
        })
        print(f"[{idx+1}/40] {r.filename} ({r.label}) -> Forehead: {forehead_ratio:.3f} | Background: {bg_ratio:.3f}")
    except Exception as e:
        print(f"Failed on {r.filename}: {e}")

df_res = pd.DataFrame(results)
df_res.to_csv("background_control.csv", index=False)

print("\n=== Control Comparison ===")
print(df_res[["forehead_ratio", "background_ratio"]].describe().T)