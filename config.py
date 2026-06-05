"""
Central configuration for the pharmaceutical continual-learning project.

Everything tweakable lives here so experiments are reproducible and you never
hard-code a path or hyper-parameter inside the logic files.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# On Colab you will typically set RAW_DIR to a folder in your mounted Drive that
# contains pill.zip and capsule.zip (see README). Locally it can be anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"          # where pill.zip / capsule.zip live (or extracted folders)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"  # clean train/val/test ImageFolders go here
OUTPUT_DIR = PROJECT_ROOT / "outputs"            # checkpoints, logs, plots

# ----------------------------------------------------------------------------
# Experiment definition
# ----------------------------------------------------------------------------
# The order in which products arrive. This is the DOMAIN-INCREMENTAL sequence:
# the model first learns "pill", then must learn "capsule" without forgetting pill.
PRODUCT_ORDER = ["pill", "capsule"]

# Binary task: every image is either normal or defective.
CLASS_NAMES = ["good", "defective"]
NUM_CLASSES = 2

# Train / val / test split ratios (applied per product, stratified by class & subtype).
SPLIT_RATIOS = (0.6, 0.2, 0.2)

SEED = 42

# ----------------------------------------------------------------------------
# Image / training hyper-parameters
# ----------------------------------------------------------------------------
IMAGE_SIZE = 224          # ResNet default input
BATCH_SIZE = 32
NUM_WORKERS = 2           # keep low on Colab
EPOCHS_PER_EXPERIENCE = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
FREEZE_BACKBONE_UNTIL = "layer3"  # freeze early ResNet blocks to fight overfitting on small data
# Set to None to fine-tune the whole network.

# Replay / regularization knobs (used by the CL strategies)
REPLAY_BUFFER_SIZE = 200
EWC_LAMBDA = 1.0
LWF_ALPHA = 1.0
LWF_TEMPERATURE = 2.0

# ImageNet normalization (because we use ImageNet-pretrained ResNet)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# ----------------------------------------------------------------------------
# VLM defect-describer (Stage 2)
# ----------------------------------------------------------------------------
# CLIP model for zero-shot defect-TYPE classification (inference only).
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# Candidate defect descriptions, per product. The CLIP describer scores a
# defective crop against these and returns the best match. Tune the wording —
# prompt engineering noticeably changes zero-shot accuracy.
DEFECT_PROMPTS = {
    "pill": {
        "color": "a pill with abnormal discoloration",
        "contamination": "a pill with surface contamination or foreign specks",
        "crack": "a pill with a crack on its surface",
        "faulty_imprint": "a pill with a faulty or smudged imprint",
        "scratch": "a pill with a scratch on its surface",
        "combined": "a pill with multiple defects",
        "pill_type": "a pill of the wrong type",
    },
    "capsule": {
        "crack": "a capsule with a crack",
        "faulty_imprint": "a capsule with a faulty imprint",
        "poke": "a capsule with a poke hole",
        "scratch": "a capsule with a scratch",
        "squeeze": "a squeezed or deformed capsule",
    },
}
