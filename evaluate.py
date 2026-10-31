import os
import numpy as np
from pipeline import extract_rgb, estimate_bpm

# Map ground-truth heart rates
TRUE_BPM = {
    "subject1": 77.3,
    "subject2": 82.6,
    "subject3": 94.7,
    "subject4": 67.0,
    "subject5": 67.8,
    "subject6": 68.8,
    "subject7": 72.2,
}

VIDEO_TEMPLATE = "data/ubfc/{}/vid.avi"
os.makedirs("cache", exist_ok=True)

rows = []
for name, truth in TRUE_BPM.items():
    rgb, fps = extract_rgb(VIDEO_TEMPLATE.format(name), f"cache/{name}.npz")
    g = estimate_bpm(rgb, fps, method="green")
    p = estimate_bpm(rgb, fps, method="pos")
    err_g = abs(g - truth)
    err_p = abs(p - truth)
    rows.append((name, truth, g, err_g, p, err_p))
    print(f"{name:8s} | Truth: {truth:.1f} | Green: {g:5.1f} (err {err_g:4.1f}) | POS: {p:5.1f} (err {err_p:4.1f})")

errs = np.array([[r[3], r[5]] for r in rows])
print("-" * 65)
print(f"MAE Green: {errs[:, 0].mean():.2f} BPM | MAE POS: {errs[:, 1].mean():.2f} BPM | Subjects: {len(rows)}")