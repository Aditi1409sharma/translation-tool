import os
import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator

# --- Setup ---
translator = GoogleTranslator(source='ja', target='en')

GT_FOLDER = "C:/Users/aditi/GT"
OUTPUT_GT_FOLDER = "C:/Users/aditi/GT_trans"
os.makedirs(OUTPUT_GT_FOLDER, exist_ok=True)

CSV_PATH = "3_lower_thresh_2/boxes_clip1.5_alpha0.9_scale2.csv"
OUTPUT_CSV_PATH = "3_lower_thresh_2/translated_boxes_clip1.5_alpha0.9_scale2.csv"

# --- Translate helper ---
def translate_text(text):
    try:
        return translator.translate(text.strip())
    except Exception as e:
        print(f"[ERROR] Failed to translate: {text} -- {e}")
        return "[ERROR]"

# --- Step 1: Translate GT files (first 100) ---
print(">>> Translating GT files (first 100)...")
gt_files = sorted([f for f in os.listdir(GT_FOLDER) if f.endswith(".txt")])[:100]

for gt_file in tqdm(gt_files):
    with open(os.path.join(GT_FOLDER, gt_file), "r", encoding="utf-8") as f:
        lines = f.readlines()

    translated_lines = [translate_text(line) for line in lines]

    with open(os.path.join(OUTPUT_GT_FOLDER, gt_file), "w", encoding="utf-8") as f:
        for line in translated_lines:
            f.write(line + "\n")

print("✅ GT translation complete and saved to:", OUTPUT_GT_FOLDER)

# --- Step 2: Translate OCR CSV ---
print(">>> Translating OCR CSV...")

df = pd.read_csv(CSV_PATH)
translated_texts = []

for text in tqdm(df["Text"]):
    translated_texts.append(translate_text(str(text)))

df["Translated_Text"] = translated_texts
df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")

print("✅ CSV translation complete and saved to:", OUTPUT_CSV_PATH)
