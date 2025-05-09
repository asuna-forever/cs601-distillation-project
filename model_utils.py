# scripts/model_utils.py
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# --- Configuration ---
# Ideally, load from config
TEACHER_MODEL_NAME = "Qwen/Qwen1.5-7B" # Or your specific Qwen model
STUDENT_MODEL_NAME = "gpt2"
CACHE_DIR = "./cache"
LOAD_IN_8BIT = False # Set to True to use bitsandbytes 8-bit quantization
LOAD_IN_4BIT = False # Set to True to use bitsandbytes 4-bit quantization (NF4)

# --- Load Student Model ---
def load_student_model(model_name=STUDENT_MODEL_NAME, cache_dir=CACHE_DIR):
    """Loads the student model (GPT-2) and its tokenizer."""
    print(f"Loading student model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

        # Add padding token if missing
        if tokenizer.pad_token is None:
            print("Adding pad token to student tokenizer")
            tokenizer.pad_token = tokenizer.eos_token
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Or use tokenizer.eos_token
            # Resize model embeddings if pad token was added
            model.config.pad_token_id = tokenizer.pad_token_id
            # model.resize_token_embeddings(len(tokenizer))
            # Ensure the model's config also reflects the pad token id
            # model.config.pad_token_id = tokenizer.pad_token_id
            print(f"Set model's pad_token_id to: {model.config.pad_token_id}")
        model_vocab_size = model.get_input_embeddings().weight.size(0)
        
        print("Student model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading student model '{model_name}': {e}")
        exit()

# --- Load Teacher Model ---
def load_teacher_model(model_name=TEACHER_MODEL_NAME, cache_dir=CACHE_DIR, load_in_8bit=LOAD_IN_8BIT, load_in_4bit=LOAD_IN_4BIT):
    """Loads the teacher model (Qwen) with optional quantization."""
    print(f"Loading teacher model: {model_name}")
    quantization_config = None
    device_map = "balanced_low_0" # Automatically distribute layers across GPUs/CPU if needed

    if load_in_8bit:
        print("Loading teacher model in 8-bit.")
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            print("Error: bitsandbytes library not found. Cannot load in 8-bit.")
            print("Install it with: pip install bitsandbytes")
            exit()
    elif load_in_4bit:
        print("Loading teacher model in 4-bit (NF4).")
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16, # Or torch.float16
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("Error: bitsandbytes library not found. Cannot load in 4-bit.")
            print("Install it with: pip install bitsandbytes")
            exit()

    try:
        # Load config first to check trust_remote_code requirement if necessary
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True) # Qwen might need trust_remote_code=True

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            quantization_config=quantization_config,
            device_map=device_map, # Handles multi-GPU/CPU placement
            trust_remote_code=True, # Qwen models often require this
            cache_dir=cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

        # Add padding token if missing (less common for newer models like Qwen, but check)
        if tokenizer.pad_token is None:
            print("Attempting to add EOS token as PAD token to teacher tokenizer.")
            # Be cautious modifying pretrained tokenizers if not needed
            # tokenizer.pad_token = tokenizer.eos_token
            # model.config.pad_token_id = tokenizer.pad_token_id
            # Consider if resizing embeddings is necessary/safe for the teacher
            # model.resize_token_embeddings(len(tokenizer))
            print("Warning: Teacher tokenizer missing pad token. Check if this is expected for Qwen.")


        print("Teacher model loaded successfully.")
        # Set to evaluation mode as it's not being trained
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading teacher model '{model_name}': {e}")
        exit()

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Loading Student ---")
    student_model, student_tokenizer = load_student_model()
    print(f"Student model type: {type(student_model)}")

    print("\n--- Loading Teacher (No Quantization) ---")
    # Note: Loading Qwen-7B without quantization requires significant VRAM (~30GB+)
    # teacher_model, teacher_tokenizer = load_teacher_model(load_in_8bit=False, load_in_4bit=False)
    # print(f"Teacher model type: {type(teacher_model)}")

    print("\n--- Loading Teacher (8-bit) ---")
    # Requires bitsandbytes
    teacher_model_8bit, teacher_tokenizer_8bit = load_teacher_model(load_in_8bit=True)
    print(f"Teacher model (8-bit) type: {type(teacher_model_8bit)}")

    # print("\n--- Loading Teacher (4-bit) ---")
    # Requires bitsandbytes
    # teacher_model_4bit, teacher_tokenizer_4bit = load_teacher_model(load_in_4bit=True)
    # print(f"Teacher model (4-bit) type: {type(teacher_model_4bit)}")

