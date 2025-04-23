from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Test conversation
user_input = "I love sci-fi movies!"
inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
reply_ids = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
reply = tokenizer.decode(reply_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
print(f"User: {user_input}")
print(f"Bot: {reply}")