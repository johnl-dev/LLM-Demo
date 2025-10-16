import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize Hugging Face model and tokenizer
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-medium")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-medium")

# Input prompt and tokenize
prompt = input("Input your prompt: ")
tokenizedPrompt = tokenizer(prompt, return_tensors='pt')

# Initialize device using CUDA cores and maps model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# copy input tensor to CUDA
tokenizedPrompt = {k: v.to(device) for k, v in tokenizedPrompt.items()}

# run inference
generated = model.generate(**tokenizedPrompt)

# output result
print("Output: "+ tokenizer.decode(generated[0], skip_special_tokens=True))