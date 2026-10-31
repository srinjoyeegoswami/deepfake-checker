import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# VIDEO_PATH = "data/ubfc/subject1/vid.avi"  #sub 1
# VIDEO_PATH = "data/ubfc/subject2/vid.avi"  #sub2
VIDEO_PATH = "data/ubfc/subject3/vid.avi"    #sub3
#     # same path you used in inspect_video.py
MAX_FRAMES = None                             # test with 300 first; set to None for the full video

TOP = [67, 109, 10, 338, 297]   # upper-forehead dots (on skin)
BROW = [105, 334]               # eyebrow dots: only used to know where the forehead ends

# --- set up the landmark detector (same as step 1) ---
base_options = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)

# --- open the video and read its real fps ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps:", fps)

rgb_means = []        # one [R, G, B] triple per frame
last = None           # last good value, used if a frame has no face
frame_idx = 0

while True:
    ok, frame_bgr = cap.read()                # read one frame; ok is False at the end of the video
    if not ok:
        break
    if MAX_FRAMES is not None and frame_idx >= MAX_FRAMES:
        break

    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        lms = result.face_landmarks[0]
        top_pts = np.array([[lms[i].x * w, lms[i].y * h] for i in TOP])
        brow_pts = np.array([[lms[i].x * w, lms[i].y * h] for i in BROW])

        x_left, x_right = top_pts[:, 0].min(), top_pts[:, 0].max()
        y_top = top_pts[:, 1].mean()          # average height of the upper row
        y_brow = brow_pts[:, 1].mean()        # average height of the eyebrows

        # trim 15% off each side (hair, temples) and stop 40% short of the eyebrows
        width = x_right - x_left
        x1 = int(x_left + 0.15 * width)
        x2 = int(x_right - 0.15 * width)
        y1 = int(y_top)
        y2 = int(y_top + 0.6 * (y_brow - y_top))

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255              # white rectangle = the forehead patch
        hull = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

        # average colour of just the masked pixels (OpenCV returns B, G, R order)
        b, g, r, _ = cv2.mean(frame_bgr, mask=mask)
        last = [r, g, b]
    # if no face was found, reuse the previous values (or zeros at the very start)
    rgb_means.append(last if last is not None else [0, 0, 0])

    # save one debug image so you can SEE the patch on frame 0
    if frame_idx == 0 and result.face_landmarks:
        debug = frame_bgr.copy()
        cv2.polylines(debug, [hull], True, (0, 255, 0), 2)

        # cv2.imwrite("patch_check_sub1.png", debug)
        # cv2.imwrite("patch_check_sub2.png", debug) #sub2
        cv2.imwrite("patch_check_sub3.png", debug)

    frame_idx += 1

cap.release()

rgb = np.array(rgb_means)                      # shape: (number of frames, 3)
print("frames processed:", len(rgb))

# np.save("subject1_rgb.npy", rgb)               # save so you don't have to re-run the slow step
# np.save("subject1_fps.npy", np.array([fps]))

# np.save("subject2_rgb.npy", rgb)               # save so you don't have to re-run the slow step
# np.save("subject2_fps.npy", np.array([fps]))

np.save("subject3_rgb.npy", rgb)               # save so you don't have to re-run the slow step
np.save("subject3_fps.npy", np.array([fps]))

# --- plot the green channel over time ---
t = np.arange(len(rgb)) / fps                  # time in seconds for each frame
plt.figure(figsize=(10, 4))
plt.plot(t, rgb[:, 1])                         # column 1 = green
plt.xlabel("time (s)")
plt.ylabel("average green of forehead patch")
plt.title("Raw green signal")

# plt.savefig("raw_green_sub1.png")

# plt.savefig("raw_green_sub2.png")

plt.savefig("raw_green_sub3.png")


# print("saved patch_check_sub1.png, raw_green_sub1.png, subject1_rgb.npy")
# print("saved patch_check_sub2.png, raw_green_sub2.png, subject2_rgb.npy")
print("saved patch_check_sub3.png, raw_green_sub3.png, subject3_rgb.npy")