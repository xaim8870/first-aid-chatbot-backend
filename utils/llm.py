import os
from huggingface_hub import InferenceClient

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")

if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY is missing. Add it to your environment variables or .env file.")

client = InferenceClient(token=HF_API_KEY)


def generate_llm_response(messages, temperature=0.2):
    try:
        response = client.chat_completion(
            model=HF_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    


    except Exception as e:
        print("❌ Hugging Face generation failed")
        raise RuntimeError(f"Hugging Face LLM request failed: {e}")