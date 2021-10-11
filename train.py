import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from fed import configuration as cfg
from fed import FedNet

import matplotlib
matplotlib.use("Agg")


def load_dataset(datasetPath):
    imagePaths = list(paths.list_images(datasetPath))
    data = []

    for imagePath in imagePaths:
        try:
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (128, 128))
            data.append(image)
        except:
            print("[WARNING] skipped {}".format(imagePath))

    return np.array(data, dtype="float32")


print("[INFO] loading data...")
fireData = load_dataset(cfg.FIRE_PATH)
nonFireData = load_dataset(cfg.NON_FIRE_PATH)

fireLabels = np.ones((fireData.shape[0],))
nonFireLabels = np.zeros((nonFireData.shape[0],))

data = np.vstack([fireData, nonFireData])
labels = np.hstack([fireLabels, nonFireLabels])
data /= 255

labels = to_categorical(labels, num_classes=2)
classTotals = labels.sum(axis=0)
classWeight = dict()

for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=cfg.TEST_SPLIT)

aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

print("[INFO] compiling model...")
opt = SGD(lr=cfg.INIT_LR, momentum=0.9, decay=cfg.INIT_LR / cfg.NUM_EPOCHS)
model = FedNet.build(width=128, height=128, depth=3, numClasses=2)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=cfg.BATCH_SIZE),
    validation_data=(testX, testY),
    steps_per_epoch=trainX.shape[0] // cfg.BATCH_SIZE,
    epochs=cfg.NUM_EPOCHS,
    class_weight=classWeight,
    verbose=1)

print("[INFO] evaluating network...")
predY = model.predict(testX, batch_size=cfg.BATCH_SIZE)
print(classification_report(testY.argmax(axis=1), predY.argmax(axis=1),
                            target_names=cfg.CLASSES))

print("[INFO] serializing network...")
model.save(cfg.MODEL_PATH)

N = np.arange(0, cfg.NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(cfg.TRAINING_PLOT_PATH)
