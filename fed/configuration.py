import os

FIRE_PATH = os.path.sep.join(["datasets", "fire"])
NON_FIRE_PATH = os.path.sep.join(["datasets", "non_fire"])

CLASSES = ["Non-Fire", "Fire"]

TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2

INIT_LR = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 50

MODEL_PATH = os.path.sep.join(["output", "fednet.model"])

TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot.png"])

SAMPLE_SIZE = 10
