from huggingface_hub import InferenceClient
import os

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_NAME = os.getenv("HF_MODEL")

client = InferenceClient(token=HF_API_KEY)

def generate_llm_response(messages, temperature=0.2):
    try:
        response = client.chat_completion(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=400,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ Hugging Face generation failed")
        raise RuntimeError(f"Hugging Face LLM request failed: {e}")