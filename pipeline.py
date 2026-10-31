import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.signal import detrend, butter, sosfiltfilt

TOP = [67, 109, 10, 338, 297]
BROW = [105, 334]

def extract_rgb(video_path, cache_path=None):
    """Extract forehead RGB averages over time, with local disk caching."""
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["rgb"], float(d["fps"])

    base = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out, last = [], None

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        h, w = frame_bgr.shape[:2]
        rgb_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )
        res = landmarker.detect(rgb_img)

        if res.face_landmarks:
            lms = res.face_landmarks[0]
            top = np.array([[lms[i].x * w, lms[i].y * h] for i in TOP])
            brow = np.array([[lms[i].x * w, lms[i].y * h] for i in BROW])

            width = top[:, 0].max() - top[:, 0].min()
            x1 = int(top[:, 0].min() + 0.15 * width)
            x2 = int(top[:, 0].max() - 0.15 * width)
            y1 = int(top[:, 1].mean())
            y2 = int(y1 + 0.6 * (brow[:, 1].mean() - y1))

            if y2 > y1 and x2 > x1:
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                b, g, r, _ = cv2.mean(frame_bgr, mask=mask)
                last = [r, g, b]

        out.append(last if last is not None else [0, 0, 0])

    cap.release()
    rgb = np.array(out)
    if cache_path:
        np.savez(cache_path, rgb=rgb, fps=fps)
    return rgb, fps

def pos_signal(rgb, fps, win_sec=1.6):
    """Plane-Orthogonal-to-Skin algorithm to cancel specular motion artifacts."""
    eps = 1e-8
    N = len(rgb)
    L = int(win_sec * fps)
    H = np.zeros(N)
    proj = np.array([[0, 1, -1],
                     [-2, 1, 1]])
    for m in range(0, N - L):
        C = rgb[m:m + L].T
        Cn = C / (C.mean(axis=1, keepdims=True) + eps)
        S = proj @ Cn
        h = S[0] + (S[0].std() / (S[1].std() + eps)) * S[1]
        H[m:m + L] += h - h.mean()
    return H

def estimate_bpm(rgb, fps, method="pos", skip_sec=0.0):
    """Compute cardiac frequency peak (BPM) from RGB signal."""
    rgb = rgb[int(skip_sec * fps):]
    sig = pos_signal(rgb, fps) if method == "pos" else rgb[:, 1]
    sig = detrend(sig)
    sos = butter(3, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")
    filt = sosfiltfilt(sos, sig)
    spec = np.abs(np.fft.rfft(filt))
    freqs = np.fft.rfftfreq(len(filt), d=1 / fps)
    band = (freqs >= 0.7) & (freqs <= 3.0)
    return freqs[band][np.argmax(spec[band])] * 60

# ---------- DFDC-style extraction: small faces in big frames ----------

def _landmarks_px(landmarker, frame, x0, y0, x1, y1, target):
    """Detect on frame[y0:y1, x0:x1], upscaled so its longest side is `target` px.
    Returns the 478 landmarks as pixel coordinates in the ORIGINAL frame, or None."""
    crop = frame[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch < 20 or cw < 20:
        return None
    s = target / max(ch, cw)
    up = cv2.resize(crop, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_CUBIC)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(up, cv2.COLOR_BGR2RGB)))
    if not res.face_landmarks:
        return None
    uh, uw = up.shape[:2]
    return np.array([[x0 + l.x * uw / s, y0 + l.y * uh / s] for l in res.face_landmarks[0]])

def _tile_search(landmarker, frame, tile=640, stride=320):
    H, W = frame.shape[:2]
    for y0 in range(0, max(H - tile, 0) + 1, stride):
        for x0 in range(0, max(W - tile, 0) + 1, stride):
            pts = _landmarks_px(landmarker, frame, x0, y0,
                                min(x0 + tile, W), min(y0 + tile, H), target=tile)
            if pts is not None:
                return pts
    return None

def extract_rgb_crop(video_path, cache_path=None):
    """Like extract_rgb, but finds the face by tiling, then tracks it with an upscaled crop.
    Frames with no valid patch are NaN (never zeros or stale values).
    Returns (rgb [frames,3] with NaNs, fps, missing_fraction)."""
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["rgb"], float(d["fps"]), float(d["missing"])

    base = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=base, num_faces=1))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out, box = [], None                          # box = (cx, cy, side) of the last face found
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        pts = None
        if box is not None:                      # 1) try a crop around where the face just was
            cx, cy, side = box
            x0, y0 = int(max(cx - side, 0)), int(max(cy - side, 0))
            x1, y1 = int(min(cx + side, W)), int(min(cy + side, H))
            pts = _landmarks_px(landmarker, frame, x0, y0, x1, y1, target=512)
        if pts is None:                          # 2) lost it: search the whole frame in tiles
            pts = _tile_search(landmarker, frame)

        row = [np.nan, np.nan, np.nan]
        if pts is not None:
            fw = pts[:, 0].max() - pts[:, 0].min()
            fh = pts[:, 1].max() - pts[:, 1].min()
            box = (pts[:, 0].mean(), pts[:, 1].mean(), 1.2 * max(fw, fh))
            top, brow = pts[TOP], pts[BROW]
            width = top[:, 0].max() - top[:, 0].min()
            x1p = int(top[:, 0].min() + 0.15 * width)
            x2p = int(top[:, 0].max() - 0.15 * width)
            y1p = int(top[:, 1].mean())
            y2p = int(y1p + 0.6 * (brow[:, 1].mean() - y1p))
            x1p, y1p = max(x1p, 0), max(y1p, 0)
            if (y2p - y1p) >= 4 and (x2p - x1p) >= 4 and y2p <= H and x2p <= W:
                b, g, r, _ = cv2.mean(frame[y1p:y2p, x1p:x2p])   # average ORIGINAL pixels
                row = [r, g, b]
        else:
            box = None
        out.append(row)
    cap.release()

    rgb = np.array(out)
    missing = float(np.isnan(rgb[:, 0]).mean())
    if cache_path:
        np.savez(cache_path, rgb=rgb, fps=fps, missing=missing)
    return rgb, fps, missing

def fill_gaps(rgb, max_missing=0.2):
    """Linearly interpolate short NaN gaps. Returns None if too much is missing."""
    bad = np.isnan(rgb[:, 0])
    if bad.mean() > max_missing or bad.all():
        return None
    idx = np.arange(len(rgb))
    out = rgb.copy()
    for c in range(3):
        out[bad, c] = np.interp(idx[bad], idx[~bad], rgb[~bad, c])
    return out