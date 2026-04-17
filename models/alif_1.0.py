from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from transformers import BitsAndBytesConfig

model_id = "large-traversaal/Alif-1.0-8B-Instruct"

# 4-bit quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer and model in 4-bit
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

# Create text generation pipeline
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# Function to chat
def chat(message):
    response = chatbot(message, max_new_tokens=100, do_sample=True, temperature=0.3)
    return response[0]["generated_text"]

# Example chat
user_input = "شہر کراچی کی کیا اہمیت ہے؟"
bot_response = chat(user_input)

print(bot_response)
