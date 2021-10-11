from tensorflow.keras.models import load_model
from fed import configuration as cfg
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default=1,
                help="predict video [1] or camera [0]")
args = vars(ap.parse_args())

print("[INFO] loading model...")
model = load_model(cfg.MODEL_PATH)

if args["video"] == 0:
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(cfg.VIDEO_PATH)

while True:
    (grabbed, frame) = camera.read()

    if args["video"] > 0 and not grabbed:
        break

    frame = imutils.resize(frame, width=800)
    frameClone = frame.copy()

    frame = cv2.resize(frame, (128, 128))
    frame = frame.astype("float32") / 255.0

    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    label = cfg.CLASSES[np.argmax(preds)]

    green = (0, 255, 0)
    red = (0, 0, 255)

    if label == "Non-Fire":
        text = "Non-Fire"
        color = green
    else:
        text = "Fire"
        color = red

    cv2.putText(frameClone, text, (35, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 5)

    cv2.imshow("Video", frameClone)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
