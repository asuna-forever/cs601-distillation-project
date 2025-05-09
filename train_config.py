# scripts/config.py
import os
import torch

# --- File Paths ---
# Path to the pre-downloaded and processed C4 subset directory (Arrow format)
# Example: "../data/c4_en_subset_50000.arrow"
# Ensure this path is correct relative to where train.py is run or use an absolute path.
C4_SUBSET_PATH = "../data/merged_training_data.arrow" # <--- ADJUST THIS PATH

# Directory to save model checkpoints and final model
OUTPUT_DIR = "./DistilQwen3"

# Directory for caching downloaded models, tokenizers, and datasets
CACHE_DIR = "./cache"

# --- Model Configuration ---
# Identifier for the teacher model on Hugging Face Hub
TEACHER_MODEL_NAME = "Qwen/Qwen3-8B" # Or "Qwen/Qwen1.5-14B", etc.

# Identifier for the student model on Hugging Face Hub
STUDENT_MODEL_NAME = "" # Or "gpt2-medium", etc.

# Model loading options (requires bitsandbytes: pip install bitsandbytes)
# Set only one of these to True if you need quantization, otherwise keep both False
LOAD_TEACHER_IN_8BIT = True # Load teacher in 8-bit precision to save memory
LOAD_TEACHER_IN_4BIT = False # Load teacher in 4-bit precision (NF4) for more memory saving

# --- Data Preprocessing ---
# Maximum sequence length for tokenization
MAX_SEQ_LENGTH = 512

# Number of workers for dataset mapping (preprocessing)
# Set to 1 if you encounter multiprocessing issues
NUM_PROC_DATA_PREP = max(1, os.cpu_count() // 2) # Use half the CPU cores, minimum 1

# --- Training Hyperparameters ---
# Number of training epochs
NUM_EPOCHS = 2

# Batch size per device (GPU/CPU) during training
# Adjust based on your GPU memory
TRAIN_BATCH_SIZE = 8
BATCH_SIZE = 8

# Learning rate for the AdamW optimizer
LEARNING_RATE = 8e-6

# Number of warmup steps for the learning rate scheduler
WARMUP_STEPS = 100

# Gradient accumulation steps
# Effective batch size = TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS
GRADIENT_ACCUMULATION_STEPS = 8 # Increase if batch size needs to be effectively larger

# Maximum norm for gradient clipping (prevents exploding gradients)
MAX_GRAD_NORM = 1.0

# Random seed for reproducibility
SEED = 42

# Mixed precision training ('fp16' or 'bf16' for faster training and less memory, None to disable)
# 'bf16' requires Ampere or newer GPUs
MIXED_PRECISION = 'fp16' # Or 'bf16' or None

# --- Distillation Parameters ---
# Weighting factor for the KL divergence distillation loss (0 <= alpha <= 1)
# Total Loss = alpha * KL_Loss + (1 - alpha) * CrossEntropy_Loss
DISTILLATION_ALPHA = 0.8 # Example: Give more weight to distillation signal

# Temperature for softening probability distributions before KL divergence calculation
# Higher temperature -> softer distributions, encourages matching broader patterns
DISTILLATION_TEMP = 2.0# Example: Moderate temperature

# --- Evaluation Configuration ---
# Batch size per device during evaluation
EVAL_BATCH_SIZE = 16

# --- Sanity Checks (Optional) ---
if LOAD_TEACHER_IN_8BIT and LOAD_TEACHER_IN_4BIT:
    raise ValueError("LOAD_TEACHER_IN_8BIT and LOAD_TEACHER_IN_4BIT cannot both be True.")

if not 0.0 <= DISTILLATION_ALPHA <= 1.0:
     raise ValueError(f"DISTILLATION_ALPHA must be between 0.0 and 1.0, but got {DISTILLATION_ALPHA}")

if DISTILLATION_TEMP <= 0:
     raise ValueError(f"DISTILLATION_TEMP must be positive, but got {DISTILLATION_TEMP}")

# --- Create Directories ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

print("Configuration loaded:")
print(f"  - C4 Subset Path: {C4_SUBSET_PATH}")
print(f"  - Output Dir: {OUTPUT_DIR}")
print(f"  - Teacher Model: {TEACHER_MODEL_NAME} (8bit: {LOAD_TEACHER_IN_8BIT}, 4bit: {LOAD_TEACHER_IN_4BIT})")
print(f"  - Student Model: {STUDENT_MODEL_NAME}")
print(f"  - Max Seq Length: {MAX_SEQ_LENGTH}")
print(f"  - Epochs: {NUM_EPOCHS}")
print(f"  - Train Batch Size: {TRAIN_BATCH_SIZE}")
print(f"  - Grad Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  - Learning Rate: {LEARNING_RATE}")
print(f"  - Distill Alpha: {DISTILLATION_ALPHA}")
print(f"  - Distill Temp: {DISTILLATION_TEMP}")
print(f"  - Mixed Precision: {MIXED_PRECISION}")
