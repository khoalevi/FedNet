import os

FIRE_PATH = os.path.sep.join(["datasets", "fire"])
NON_FIRE_PATH = os.path.sep.join(["datasets", "non_fire"])

VIDEO_PATH = os.path.sep.join(["examples", "lighter.mp4"])

CLASSES = ["Non-Fire", "Fire"]

TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2

INIT_LR = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 50

MODEL_PATH = os.path.sep.join(["output", "fednet.model"])

TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot.png"])

SAMPLE_SIZE = 10

DETECT_IMAGES = os.path.sep.join(["datasets", "detection", "images"])
DETECT_ANNOTS = os.path.sep.join(["datasets", "detection", "annotations"])

GEN_PATH = os.path.sep.join(["datasets", "generation"])
GEN_FIRE_PATH = os.path.sep.join([GEN_PATH, "fire"])
GEN_NON_FIRE_PATH = os.path.sep.join([GEN_PATH, "non_fire"])

# define the number of max proposals used when running selective
# search for gathering training data and performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# define the maximum number of fire and non fire images
# to be generated from each image
MAX_FIRE = 30
MAX_NON_FIRE = 10

INPUT_DIMS = (224, 224)

DETECTOR_PATH = os.path.sep.join(["output", "fire_detector.h5"])
ENCODER_PATH = os.path.sep.join(["output", "fire_encoder.pickle"])

# define the minimum probability required for a positive prediction
MIN_PROBA = 0.99