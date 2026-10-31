import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DIR = "data/dfdc/train_sample_videos/"
base = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(base_options=base, num_faces=1))

def find_face_in_tiles(frame, tile=640, stride=320):
    """Slide a tile over the big frame; return the face box (x1,y1,x2,y2) in full-frame pixels, or None."""
    H, W = frame.shape[:2]
    for y0 in range(0, max(H - tile, 0) + 1, stride):
        for x0 in range(0, max(W - tile, 0) + 1, stride):
            crop = frame[y0:y0 + tile, x0:x0 + tile]
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                             data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            if res.face_landmarks:
                ch, cw = crop.shape[:2]
                xs = [l.x * cw + x0 for l in res.face_landmarks[0]]
                ys = [l.y * ch + y0 for l in res.face_landmarks[0]]
                return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    return None

def probe(fname, n_probe=8):
    cap = cv2.VideoCapture(DIR + fname)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    found, widths, saved = 0, [], False
    for idx in np.linspace(0, total - 1, n_probe).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        box = find_face_in_tiles(frame)
        if box:
            found += 1
            widths.append(box[2] - box[0])
            if not saved:                                   # save one frame with the box drawn
                cv2.rectangle(frame, box[:2], box[2:], (0, 255, 0), 4)
                cv2.imwrite(f"tiles_{fname}.png", frame)
                saved = True
    cap.release()
    w = f"{np.median(widths):.0f}px" if widths else "n/a"
    print(f"{fname}: face found in tiles {found}/{n_probe} | median face width {w}")

for f in ["bejhvclboh.mp4", "abofeumbvv.mp4", "acifjvzvpm.mp4", "acqfdwsrhi.mp4"]:
    probe(f)