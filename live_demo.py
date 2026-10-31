import collections
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt
from pipeline import pos_signal

TOP, BROW = [67, 109, 10, 338, 297], [105, 334]
WINDOW_SEC, MIN_SEC = 20, 10  # keep 20 s of history; start reporting after 10 s

base = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
)

def estimate(buf):
    arr = np.array(buf)  # columns: t, r, g, b
    span = arr[-1, 0] - arr[0, 0]
    fps = (len(arr) - 1) / max(span, 1e-6)  # real webcam fps, measured from timestamps
    sig = detrend(pos_signal(arr[:, 1:4], fps))
    sos = butter(3, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")
    filt = sosfiltfilt(sos, sig)
    spec = np.abs(np.fft.rfft(filt))
    f = np.fft.rfftfreq(len(filt), d=1 / fps)
    band = (f >= 0.7) & (f <= 3.0)
    return f[band][np.argmax(spec[band])] * 60, filt

# Robust camera opening for Windows (DirectShow backend + index fallback)
cap = None
for idx in [0, 1, 2]:
    temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if temp_cap.isOpened():
        ret, probe_frame = temp_cap.read()
        if ret and probe_frame is not None:
            cap = temp_cap
            print(f"Connected to webcam at index {idx} via CAP_DSHOW")
            break
        temp_cap.release()

if cap is None:
    # Fallback to default backend if DirectShow was not required
    for idx in [0, 1]:
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            ret, probe_frame = temp_cap.read()
            if ret and probe_frame is not None:
                cap = temp_cap
                print(f"Connected to webcam at index {idx} via default backend")
                break
            temp_cap.release()

if cap is None:
    raise RuntimeError(
        "Could not access webcam. Verify that your camera is not in use by another app "
        "and Windows Settings -> Privacy & Security -> Camera permissions are enabled."
    )

buf, bpm, filt, last_calc, t0 = collections.deque(), None, None, 0, time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("Dropped camera frame, retrying...")
        continue
    frame = cv2.flip(frame, 1)  # mirror view feels natural
    h, w = frame.shape[:2]
    now = time.time() - t0

    res = landmarker.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
    )
    face_ok = False
    if res.face_landmarks:
        lms = res.face_landmarks[0]
        top = np.array([[lms[i].x * w, lms[i].y * h] for i in TOP])
        brow = np.array([[lms[i].x * w, lms[i].y * h] for i in BROW])
        width = top[:, 0].max() - top[:, 0].min()
        x1 = int(top[:, 0].min() + 0.15 * width)
        x2 = int(top[:, 0].max() - 0.15 * width)
        y1 = int(top[:, 1].mean())
        y2 = int(y1 + 0.6 * (brow[:, 1].mean() - y1))
        
        # Ensure coordinates stay within frame bounds
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if y2 - y1 >= 4 and x2 - x1 >= 4:
            b, g, r, _ = cv2.mean(frame[y1:y2, x1:x2])
            buf.append((now, r, g, b))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_ok = True

    while buf and now - buf[0][0] > WINDOW_SEC:  # drop samples older than the window
        buf.popleft()

    if len(buf) > 30 and now - last_calc > 1.0 and buf[-1][0] - buf[0][0] >= MIN_SEC:
        bpm, filt = estimate(buf)
        last_calc = now

    # --- on-screen text ---
    if bpm is not None:
        txt = f"POS estimate: {bpm:.0f} bpm"
    elif face_ok:
        elapsed = buf[-1][0] - buf[0][0] if buf else 0
        txt = f"collecting... {elapsed:.0f}/{MIN_SEC} s (stay still)"
    else:
        txt = "no face found"
        
    cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(
        frame,
        "learning project, not a medical device",
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # --- mini waveform of the filtered pulse signal ---
    if filt is not None and len(filt) > 2:
        seg = filt[-300:]
        ys = (h - 80) - ((seg - seg.min()) / (np.ptp(seg) + 1e-9)) * 60
        xs = np.linspace(20, w - 20, len(seg))
        cv2.polylines(frame, [np.stack([xs, ys], 1).astype(np.int32)], False, (0, 255, 0), 2)

    cv2.imshow("pulse demo (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()