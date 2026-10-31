import cv2

path = "data/ubfc/subject1/vid.avi"   # change to your file name
cap = cv2.VideoCapture(path)

fps = cap.get(cv2.CAP_PROP_FPS)                    # frames per second
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # total frames
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("fps:", fps, "| frames:", frames, "| size:", w, "x", h)
print("length in seconds:", frames / fps)

ok, frame = cap.read()                             # read the first frame
cv2.imwrite("first_frame.png", frame)              # save it so you can look at it
cap.release()