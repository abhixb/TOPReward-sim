"""AWR-specific configuration. Parent repo config imported where needed."""

from pathlib import Path

# ─── Dataset ───
DATASET_REPO_ID = ""               # e.g. "your_username/so100_pick_cube" or local path
DATASET_ROOT = None                # Set if using a local dataset path
CAMERA_KEY = "observation.image"   # Key for RGB frames in LeRobot dataset
INSTRUCTION = "Pick up the red cube and place it in the box"

# ─── Scoring ───
NUM_EVAL_FRAMES = 20               # Prefix evaluations per episode
MAX_FRAMES_PER_PREFIX = 16         # Max frames sent to VLM per prefix

# ─── AWR ───
TAU = 2.0                          # Scaling factor for advantage weights (paper default)
DELTA_MAX = 2.0                    # Max allowed weight (paper default)
SUBTRACT_MEAN = True               # Subtract dataset mean progress before computing weights
WEIGHT_CLIP_MIN = 0.1              # Don't let any weight go below this (prevents zero-gradient)

# ─── Training ───
POLICY_TYPE = "act"                # "act" or "diffusion"
BASE_CHECKPOINT = ""               # Path to pretrained policy checkpoint (or HuggingFace repo)
LEARNING_RATE = 1e-5               # Lower than standard BC since we're fine-tuning
NUM_TRAIN_STEPS = 5000             # AWR fine-tuning steps
BATCH_SIZE = 8
EVAL_EVERY = 500
SAVE_EVERY = 1000
DEVICE = "cuda:0"

# ─── Output ───
OUTPUT_DIR = Path(__file__).parent / "outputs"
