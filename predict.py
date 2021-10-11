from tensorflow.keras.models import load_model
from fed import configuration as cfg
from imutils import paths
import numpy as np
import imutils
import random
import cv2

print("[INFO] loading model...")
model = load_model(cfg.MODEL_PATH)

print("[INFO] predicting...")
firePaths = list(paths.list_images(cfg.FIRE_PATH))
nonFirePaths = list(paths.list_images(cfg.NON_FIRE_PATH))

imagePaths = firePaths + nonFirePaths
random.shuffle(imagePaths)
imagePaths = imagePaths[:cfg.SAMPLE_SIZE]

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    output = image.copy()

    image = cv2.resize(image, (128, 128))
    image = image.astype("float32") / 255.0

    preds = model.predict(np.expand_dims(image, axis=0))[0]
    label = cfg.CLASSES[np.argmax(preds)]

    text = label if label == "Non-Fire" else "Fire"
    output = imutils.resize(output, width=500)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 5)
    
    cv2.imshow("", output)
    cv2.waitKey(0)
