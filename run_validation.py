import sys
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from distill_model import DistilQwen3ForCausalLM
from datasets import load_dataset
tokenizer_name="Qwen/Qwen3-8B"
model_name = "./DistilQwen3finetune/final_model" # Path to the fine-tuned model directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,trust_remote_code=True)
model = DistilQwen3ForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()  
dataset = load_dataset("dogtooth/default_project_dev_test", split="dev_test")

total_loss = 0.0
total_tokens = 0
i=0
for example in dataset:
    i+=1
    text = example["text"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True,max_length=1024)
    input_ids = inputs.input_ids.to(device) 

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss

    num_tokens = input_ids.numel()
    total_loss += loss.item() * num_tokens
    total_tokens += num_tokens
    print(f"Example {i}: Loss: {loss.item()}, Tokens: {num_tokens}")
avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)
print("Perplexity:", perplexity)