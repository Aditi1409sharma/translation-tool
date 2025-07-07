# translate_utils.py
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-ja-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)


def translate_text(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return ""
        inputs = tokenizer([text.strip()], return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[ERROR] Translation failed: {e}")
        return "[ERROR]"
