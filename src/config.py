import os

#Root directory of the project (this file's parent directory's parent)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")

#Training settings
NUM_EPOCHS_BASELINE = 2      # short run for "old" model
NUM_EPOCHS_FINETUNE = 5      # longer fine-tune
BATCH_SIZE = 32
LEARNING_RATE_BASELINE = 1e-3
LEARNING_RATE_FINETUNE = 1e-4

# How many countries/classes
DEFAULT_NUM_CLASSES = 50

# Device selection
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
