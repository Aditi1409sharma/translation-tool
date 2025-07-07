# import os
# import pandas as pd
# from tqdm import tqdm
# from deep_translator import GoogleTranslator

# # --- Setup ---
# translator = GoogleTranslator(source='ja', target='en')

# GT_FOLDER = "C:/Users/aditi/GT"
# OUTPUT_GT_FOLDER = "C:/Users/aditi/GT_trans"
# os.makedirs(OUTPUT_GT_FOLDER, exist_ok=True)

# CSV_PATH = "3_lower_thresh_2_500/boxes_clip1.5_alpha0.9_scale2.csv"
# OUTPUT_CSV_PATH = "3_lower_thresh_2_500/translated_boxes_clip1.5_alpha0.9_scale2.csv"

# from transformers import MarianMTModel, MarianTokenizer

# # Load model only once
# model_name = 'Helsinki-NLP/opus-mt-ja-en'
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# model = MarianMTModel.from_pretrained(model_name)

# def translate_text(text):
#     try:
#         if not text.strip():
#             return ""
#         tokens = tokenizer(["こんにちは", "お元気ですか？"], return_tensors="pt", padding=True, truncation=True)

#         translation = model.generate(**tokens)
#         return tokenizer.decode(translation[0], skip_special_tokens=True)
#     except Exception as e:
#         print(f"[ERROR] {text} --> {e}")
#         return "[ERROR]"


# # --- Step 1: Translate GT files (first 100) ---
# print(">>> Translating GT files (first 100)...")
# gt_files = sorted([f for f in os.listdir(GT_FOLDER) if f.endswith(".txt")])[:500]

# for gt_file in tqdm(gt_files):
#     with open(os.path.join(GT_FOLDER, gt_file), "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     translated_lines = [translate_text(line) for line in lines]

#     with open(os.path.join(OUTPUT_GT_FOLDER, gt_file), "w", encoding="utf-8") as f:
#         for line in translated_lines:
#             f.write(line + "\n")

# print("✅ GT translation complete and saved to:", OUTPUT_GT_FOLDER)

# # --- Step 2: Translate OCR CSV ---
# print(">>> Translating OCR CSV...")

# df = pd.read_csv(CSV_PATH)
# translated_texts = []

# for text in tqdm(df["Text"]):
#     translated_texts.append(translate_text(str(text)))

# df["Translated_Text"] = translated_texts
# df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")

# print("✅ CSV translation complete and saved to:", OUTPUT_CSV_PATH)




import os
import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

# --- Setup model ---
model_name = 'Helsinki-NLP/opus-mt-ja-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# --- Translation function ---
def translate_text(text):
    try:
        if not isinstance(text, str) or not text.strip():
            return ""
        inputs = tokenizer([text.strip()], return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[ERROR] Could not translate: {text} --> {e}")
        return "[ERROR]"

# --- Paths ---
GT_FOLDER = "C:/Users/aditi/GT"
OUTPUT_GT_FOLDER = "C:/Users/aditi/GT_trans"
os.makedirs(OUTPUT_GT_FOLDER, exist_ok=True)

CSV_PATH = "best_version/boxes_clip1.5_alpha0.9_scale2.csv"
OUTPUT_CSV_PATH = "best_version/translated_boxes_clip1.5_alpha0.9_scale2.csv"

# --- Step 1: Translate GT .txt files ---
print(">>> Translating GT files (up to 50)...")
gt_files = sorted([f for f in os.listdir(GT_FOLDER) if f.endswith(".txt")])[:500]

for gt_file in tqdm(gt_files, desc="Translating GT files"):
    with open(os.path.join(GT_FOLDER, gt_file), "r", encoding="utf-8") as f:
        lines = f.readlines()

    translated_lines = [translate_text(line) for line in lines]

    with open(os.path.join(OUTPUT_GT_FOLDER, gt_file), "w", encoding="utf-8") as f:
        for line in translated_lines:
            f.write(line + "\n")

print("✅ GT file translation complete and saved to:", OUTPUT_GT_FOLDER)

# --- Step 2: Translate OCR CSV ---
print(">>> Translating merged OCR CSV...")

df = pd.read_csv(CSV_PATH)

if "Text" not in df.columns:
    raise ValueError(f"Column 'Text' not found in {CSV_PATH}. Ensure OCR output is correctly formatted.")

df["Translated_Text"] = [
    translate_text(str(txt)) if isinstance(txt, str) and txt.strip() else ""
    for txt in tqdm(df["Text"], desc="Translating CSV phrases")
]

df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
print("✅ OCR CSV translation saved to:", OUTPUT_CSV_PATH)

