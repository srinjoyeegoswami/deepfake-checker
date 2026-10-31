import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

TOP = [67, 109, 10, 338, 297]
BROW = [105, 334]
DIR = "data/dfdc/train_sample_videos/"

base = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(base_options=base, num_faces=1))

def probe(fname, n_probe=15):
    cap = cv2.VideoCapture(DIR + fname)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    found = valid = 0
    first_found_frame = None
    info = ""
    for idx in np.linspace(0, total - 1, n_probe).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if not res.face_landmarks:
            continue
        found += 1
        lms = res.face_landmarks[0]
        top = np.array([[lms[i].x * w, lms[i].y * h] for i in TOP])
        brow = np.array([[lms[i].x * w, lms[i].y * h] for i in BROW])
        width = top[:, 0].max() - top[:, 0].min()
        x1, x2 = top[:, 0].min() + 0.15 * width, top[:, 0].max() - 0.15 * width
        y1 = top[:, 1].mean()
        y2 = y1 + 0.6 * (brow[:, 1].mean() - y1)
        if y2 > y1 and x2 > x1:
            valid += 1
        if first_found_frame is None:
            first_found_frame = frame
            info = f"patch y1={y1:.0f} y2={y2:.0f} x1={x1:.0f} x2={x2:.0f}, face width ~{width:.0f}px"
    if first_found_frame is None:                       # save a raw frame so you can look at it
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, fr = cap.read()
        if ok:
            cv2.imwrite(f"probe_{fname}.png", fr)
    cap.release()
    print(f"{fname}: frame {w}x{h} | face found {found}/{n_probe} | valid patch {valid}/{n_probe} | {info}")

failed = ["bejhvclboh.mp4", "abofeumbvv.mp4", "acifjvzvpm.mp4", "acqfdwsrhi.mp4"]
df = pd.read_csv("sanity_check.csv")
worked = df[df["ratio"].notna()]["file"].head(3).tolist()     # a few clips that succeeded, for comparison

print("--- FAILED clips ---")
for f in failed: probe(f)
print("--- WORKED clips ---")
for f in worked: probe(f)