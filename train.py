# scripts/train.py
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM, # We'll load student/teacher via model_utils
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling, # Useful collator
    set_seed,
    AutoTokenizer
)
import pandas as pd
from accelerate import Accelerator # Handles device placement, mixed precision
from tqdm.auto import tqdm # Progress bars
import evaluate # For metrics
import train_config
from train_config import *

# Import local modules
from model_utils import load_student_model, load_teacher_model
from distillation_loss import DistillationLoss


import os
from datasets import load_from_disk, Dataset ,load_dataset # Modified import
# from transformers import AutoTokenizer # Keep Tokenizer loading
# import itertools # No longer needed

try:
    from distill_model import (
        DistilQwen3Config,
        DistilQwen3ForCausalLM,
        # You might not need to import lower-level classes like
        # Qwen3ModelSharedLayers here unless used directly elsewhere.
        # The DistilQwen3ForCausalLM class should handle their instantiation.
    )
    print("Successfully imported custom DistilQwen3 model classes.")
except ImportError as e:
    print(f"Error importing custom model classes: {e}")
    print("Please ensure 'distill_model.py' exists and contains the necessary class definitions.")
    exit()

# --- Configuration ---
# # Point C4_SUBSET_PATH to your uploaded Arrow data directory
C4_SUBSET_PATH = "/data/you/cs601-LLM-project-2/data/RedPajama-Data-60000.arrow"
# C4_SUBSET_PATH = "/data/you/cs601-LLM-project-2/data/c4_en_subset_10000.arrow"  # <--- Modify here! Relative or absolute path to the data directory on the server
# C4_SUBSET_PATH = "../data/merged-DailyDialog-SQuAD-CNN-9000.arrow"  # <--- Modify here! Relative or absolute path to the data directory on the server
# # STREAMING_SUBSET_SIZE = 10000 # No longer needed
# TEACHER_MODEL_NAME = "Qwen/Qwen3-8B" # Or your specific Qwen model
# # STUDENT_MODEL_NAME = "Qwen/Qwen3-1.7B"
# MAX_SEQ_LENGTH = 256
# CACHE_DIR = "./cache"

# It's crucial that the student uses the same vocabulary as the teacher
print(f"Loading tokenizer from TEACHER model: {TEACHER_MODEL_NAME}")
try:
    # Load tokenizer associated with the teacher model
    student_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME, cache_dir=CACHE_DIR)
    if student_tokenizer.pad_token is None:
        # Add padding token if missing, common for decoder models
        student_tokenizer.pad_token = student_tokenizer.eos_token
        print("Added EOS token as PAD token to tokenizer.")
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer for {TEACHER_MODEL_NAME}: {e}")
    exit()


# --- Data Loading Function ---
def get_c4_subset(subset_path=C4_SUBSET_PATH, max_samples=None):
    """Loads a pre-downloaded C4 subset (Arrow format) from local disk, optionally limiting the number of samples loaded."""
    if subset_path and os.path.exists(subset_path):
        print(f"Loading C4 subset from local disk: {subset_path}")
        try:
            # Use load_from_disk to load the dataset directory in Arrow format
            loaded_dataset = load_from_disk(subset_path)

            # Handle the case of DatasetDict
            if not isinstance(loaded_dataset, Dataset):
                if 'train' in loaded_dataset:
                    print("Extracting 'train' split from the loaded DatasetDict.")
                    loaded_dataset = loaded_dataset['train']
                else:
                    raise TypeError(f"Data type loaded from disk is not the expected Dataset or DatasetDict, but {type(loaded_dataset)}")

            # Truncate to the first max_samples samples (if specified)
            if max_samples is not None:
                loaded_dataset = loaded_dataset.select(range(min(max_samples, len(loaded_dataset))))
                print(f"Successfully loaded the first {len(loaded_dataset)} samples.")
            else:
                print(f"Successfully loaded {len(loaded_dataset)} samples.")

            return loaded_dataset

        except Exception as e:
            print(f"Error loading dataset from disk ({subset_path}): {e}")
            print("Please ensure the path is correct and the directory contains valid Arrow data files.")
            exit()
    else:
        print(f"Error: Specified C4 subset path does not exist: {subset_path}")
        print("Please ensure you have uploaded the downloaded subset directory to the correct location on the server.")
        exit()


# --- Preprocessing Function  ---
def preprocess_function(examples):
    """Tokenizes the text data."""
    tokenized_inputs = student_tokenizer(
        examples["text"],
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors=None,
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"][:]
    return tokenized_inputs

# --- Main Data Preparation  ---
def prepare_data(num_proc=4):
    """Loads, preprocesses, and splits the dataset."""
    print("Loading C4 subset from local disk...") # Update log message
    raw_dataset = get_c4_subset() # Now loads from local disk
    # raw_dataset = load_dataset("dogtooth/default_project_dev_test", split="dev_test")
    print("Preprocessing dataset...")
    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=raw_dataset.column_names
    )
    print("length of every training data:", len(tokenized_dataset[0]["input_ids"]))
    print("max token id", max(tokenized_dataset[0]["input_ids"]))
    print("length of tokenizer vocab:", len(student_tokenizer))
    print("vocab size:", student_tokenizer.vocab_size)
    print("Splitting dataset (90% train, 10% validation)...")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    print("Data preparation complete.")
    print("Train dataset size:", len(split_dataset['train']))
    print("Validation dataset size:", len(split_dataset['test']))

    return split_dataset['train'], split_dataset['test'], student_tokenizer

# --- Configuration ---
# Ideally load from config.py or command-line args
OUTPUT_DIR = "./DistilQwen3finetune2"
LOAD_TEACHER_IN_8BIT = True # Adjust based on your hardware
LOAD_TEACHER_IN_4BIT = False # Adjust based on your hardware

# # Training Hyperparameters
# NUM_EPOCHS = 1
# BATCH_SIZE = 8 # Adjust based on GPU memory
# LEARNING_RATE = 1e-6
# WARMUP_STEPS = 50
# DISTILLATION_ALPHA = 0.8 # Weight for KL loss
# DISTILLATION_TEMP = 1.2 # Temperature for KL loss
# GRADIENT_ACCUMULATION_STEPS = 1 # Increase for larger effective batch size
# MAX_GRAD_NORM = 1.0 # For gradient clipping
# SEED = 42

# --- Initialization ---
set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Accelerator
# mixed_precision='fp16' or 'bf16' can speed up training and save memory
accelerator = Accelerator(gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, mixed_precision='fp16')

# --- Load Data ---
print("Preparing data...")
# Use num_proc=1 if running into issues with multiprocessing and datasets/accelerate
train_dataset, eval_dataset, student_tokenizer = prepare_data(num_proc=1)
print(f"Train dataset size: {len(train_dataset)}")
print(f"length of every sample in train dataset: {len(train_dataset[0]['input_ids'])}")
# Data Collator - handles padding dynamically per batch if needed
# We already padded to max_length in preprocessing, but this is standard practice
data_collator = DataCollatorForLanguageModeling(tokenizer=student_tokenizer, mlm=False)
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=BATCH_SIZE)


print("Initializing custom DistilQwen3 student model...")
# 1. Create the configuration object for your custom model
# Ensure vocab_size matches the loaded tokenizer
student_config = DistilQwen3Config(
    # vocab_size=len(student_tokenizer), # IMPORTANT: Match tokenizer vocab size
    # Set other distilled model parameters (layers, hidden size, bottleneck etc.)
    # These defaults are defined in your DistilQwen3Config class,
    # override them here if needed, e.g.:
    # hidden_size=3072,
    # num_hidden_layers=24,
    # mlp_bottleneck_dim=1536,
    # num_key_value_heads=8,
)
print(f"Using student config: {student_config}")

# 2. Instantiate your custom model using the configuration
# This model will have random weights initially.
try:
    # student_model = DistilQwen3ForCausalLM(student_config)
    student_model = DistilQwen3ForCausalLM.from_pretrained("./DistilQwen3finetune/final_model")
    print("Custom student model initialized successfully.")
except Exception as e:
    print(f"Error initializing custom student model: {e}")
    exit()
print("student vocab size:", student_tokenizer.vocab_size)
print("student vocab size:", student_model.config.vocab_size)

# --- Load Models ---
print("Loading teacher model...")
teacher_model, _ = load_teacher_model(TEACHER_MODEL_NAME, load_in_8bit=LOAD_TEACHER_IN_8BIT, load_in_4bit=LOAD_TEACHER_IN_4BIT)
print("teacher vocab size:", teacher_model.config.vocab_size)

# Ensure teacher model is on the correct device(s) if not using device_map="auto" effectively
# teacher_model = teacher_model.to(accelerator.device) # Usually handled by device_map

# --- Setup Optimizer, Scheduler, Loss ---
optimizer = AdamW(student_model.parameters(), lr=LEARNING_RATE)

num_training_steps = (len(train_dataloader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=num_training_steps,
)

# Use standard CrossEntropyLoss for the student's base loss
# ignore_index=-100 is common if padding tokens should be ignored

distill_loss_fct = DistillationLoss(
    base_criterion=torch.nn.CrossEntropyLoss(ignore_index=-100),
    teacher_model=teacher_model,
    # student_model=student_model, # Pass student model
    student_tokenizer=student_tokenizer, # Pass tokenizer
    alpha=train_config.DISTILLATION_ALPHA # Use config
)

# --- Prepare with Accelerator ---
# Prepare student model, optimizer, dataloaders, scheduler
# Crucially, prepare the projection layer inside DistillationLoss if it exists
if False: # This conditional logic for projection layer preparation seems to be a placeholder or for experimentation
    # Prepare the projection layer along with the student model
    student_model, distill_loss_fct.projection, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        student_model, distill_loss_fct.projection, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    print("Prepared student model and projection layer.")
else:
    # Only prepare the student model
    student_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        student_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    print("Prepared student model (no projection layer).")

# Teacher model is not prepared as it's not trained
# Ensure teacher model is on the correct device if not handled by device_map
# teacher_model = teacher_model.to(accelerator.device) # Might be needed depending on loading strategy

# --- Evaluation Metric ---
perplexity_metric = evaluate.load("perplexity", module_type="metric", cache_dir=train_config.CACHE_DIR) # Use config
all_losses = []
# --- Training Loop ---
print("Starting training...")
progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0

for epoch in range(train_config.NUM_EPOCHS): # Use config
    student_model.train() # Set student model to training mode
    # if distill_loss_fct.needs_projection:
    #     distill_loss_fct.projection.train() # Ensure projection layer is also in train mode

    total_loss_epoch = 0.0
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(student_model): # Handles gradient accumulation

            student_outputs = student_model(**batch)

            # Calculate loss using hidden states
            loss, loss_dict = distill_loss_fct(
                student_outputs=student_outputs,
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_mask=batch["attention_mask"] # Pass attention mask
            )

            total_loss_epoch += loss.item() / train_config.GRADIENT_ACCUMULATION_STEPS

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                 # Unscale gradients before clipping
                 # accelerator.unscale_gradients(optimizer) # Usually handled by accelerator.clip_grad_norm_
                 accelerator.clip_grad_norm_(student_model.parameters(), train_config.MAX_GRAD_NORM) # Use config

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Update progress bar and logging
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1
            loss_log = {
                "step": completed_steps,
                "epoch": epoch,
                "total_loss": loss.item() / train_config.GRADIENT_ACCUMULATION_STEPS, # Record average loss
                "loss_ce": loss_dict['loss_ce'].item(),
                "loss_kl": loss_dict['loss_kl'].item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            # Log only on the main process
            if accelerator.is_local_main_process:
                all_losses.append(loss_log)
            # --- **End of logging** ---

            progress_bar.set_postfix(loss_log)

   
    # --- Save Model Checkpoint ---
    if accelerator.is_local_main_process: # Only save on the main process
        save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch}")
        accelerator.wait_for_everyone() # Ensure all processes are finished before saving
        unwrapped_model = accelerator.unwrap_model(student_model) # Get the underlying model
        unwrapped_model.save_pretrained(save_path)
        student_tokenizer.save_pretrained(save_path)
        accelerator.print(f"Model checkpoint saved to {save_path}")


print("Training finished.")
if accelerator.is_local_main_process:
    if all_losses:
        loss_df = pd.DataFrame(all_losses)
        loss_csv_path = os.path.join(train_config.OUTPUT_DIR, "training_losses.csv")
        loss_df.to_csv(loss_csv_path, index=False)
        accelerator.print(f"Training losses saved to {loss_csv_path}")
    else:
        accelerator.print("No losses were recorded (maybe training failed early?).")
# --- Final Save ---
if accelerator.is_local_main_process:
    final_save_path = os.path.join(OUTPUT_DIR, "final_model")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(student_model)
    unwrapped_model.save_pretrained(final_save_path)
    student_tokenizer.save_pretrained(final_save_path)
    accelerator.print(f"Final model saved to {final_save_path}")