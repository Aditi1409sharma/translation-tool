import os
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

# --- Paths ---
GT_DIR = 'C:/Users/aditi/GT'
GT_TRANSLATED_DIR = 'C:/Users/aditi/GT_translated'

# Path to local model
MODEL_PATH = 'C:/Users/aditi/translation_models/opus-mt-ja-en'

# --- Create output folder ---
os.makedirs(GT_TRANSLATED_DIR, exist_ok=True)

# --- Load Helsinki-NLP Japanese → English model (local path) ---
tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
model = MarianMTModel.from_pretrained(MODEL_PATH)

# --- List all GT files ---
gt_files = [f for f in os.listdir(GT_DIR) if f.endswith('.txt')]

# --- Process each GT file ---
for fname in tqdm(gt_files, desc='Translating GT files'):
    in_path = os.path.join(GT_DIR, fname)
    out_path = os.path.join(GT_TRANSLATED_DIR, fname)

    with open(in_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            parts = line.strip().split(',')
            if len(parts) < 10:
                continue  # skip malformed line

            lang = parts[8]
            text = parts[9]

            if lang.lower() == 'japanese' and text.strip():
                # Translate Japanese text
                inputs = tokenizer([text], return_tensors="pt", truncation=True)
                translated = model.generate(**inputs)
                en_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                parts[9] = en_text

            # Write updated line
            fout.write(','.join(parts) + '\n')

print('✅ All translations saved in:', GT_TRANSLATED_DIR)
