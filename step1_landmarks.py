import cv2                                    # OpenCV: read/write images, draw shapes
import mediapipe as mp                        # MediaPipe: the face landmark model
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# 1. Load the model file you downloaded (must be in the same folder as this script)
base_options = mp_python.BaseOptions(model_asset_path="face_landmarker.task")

# 2. Settings: we only expect one face in the video
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)

# 3. Create the landmark detector from those settings
landmarker = vision.FaceLandmarker.create_from_options(options)

# 4. Read the frame you saved earlier. OpenCV loads colours in BGR order
frame_bgr = cv2.imread("first_frame.png")
if frame_bgr is None:                         # imread returns None if the file isn't found
    raise SystemExit("Could not read first_frame.png. Check the file path.")

# 5. MediaPipe expects RGB order, so convert
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# 6. Wrap the array in MediaPipe's image type
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

# 7. Run the detector
result = landmarker.detect(mp_image)

# 8. Stop if no face was found
if not result.face_landmarks:
    raise SystemExit("No face found in the frame.")

# 9. Take the first face: a list of 478 points
landmarks = result.face_landmarks[0]
print("number of landmarks:", len(landmarks))

# 10. Landmark coordinates are fractions (0 to 1) of the image size.
#     Multiply by width/height to get pixel positions.
h, w = frame_bgr.shape[:2]

# 11. Draw every point as a tiny green dot
all_points = frame_bgr.copy()
for lm in landmarks:
    x, y = int(lm.x * w), int(lm.y * h)
    cv2.circle(all_points, (x, y), 1, (0, 255, 0), -1)
cv2.imwrite("landmarks_all.png", all_points)

# 12. Draw candidate forehead points with their index numbers.
#     These are common forehead/brow indices in MediaPipe's numbering, so check them visually.
candidates = [10, 67, 109, 338, 297, 151, 9, 107, 66, 105, 336, 296, 334]
labelled = frame_bgr.copy()
for i in candidates:
    lm = landmarks[i]
    x, y = int(lm.x * w), int(lm.y * h)
    cv2.circle(labelled, (x, y), 3, (0, 0, 255), -1)
    cv2.putText(labelled, str(i), (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
cv2.imwrite("landmarks_forehead.png", labelled)

print("saved landmarks_all.png and landmarks_forehead.png")