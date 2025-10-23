import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize accelerator for distributed inference
accelerator = Accelerator()

# Initialize Hugging Face model and tokenizer
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")

# Prepare model for distributed inference
model = accelerator.prepare(model)

# Input prompt and tokenize
prompt = input("Input your prompt: ")
tokenizedPrompt = tokenizer(prompt, return_tensors='pt')

# copy input tensor to CUDA
tokenizedPrompt = {k: v.to(accelerator.device) for k, v in tokenizedPrompt.items()}

# run inference
generated = model.generate(**tokenizedPrompt)

# output result
print("Output: "+ tokenizer.decode(generated[0], skip_special_tokens=True))