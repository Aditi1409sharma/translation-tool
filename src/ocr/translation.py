import os
import cv2
import numpy as np
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm
from ocr_infer import preprocess_image, ocr  # Your existing PaddleOCR wrapper
from grid_search import to_gray, apply_clahe, denoise, sharpen, upscale  # Replace if needed

# Translation model
MODEL_NAME = "Helsinki-NLP/opus-mt-ja-en"
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)

# Paths
IMAGE_DIR = "C:/Users/aditi/ImagesPart2"
OUTPUT_DIR = "translated_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Preprocessing config
clip = 1.8
alpha = 0.9

# Check for Japanese text
def is_japanese(text):
    return any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9faf' for ch in text)

# Translate Japanese text to English
def translate_text(text):
    batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    translated = model.generate(**batch)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Run pipeline
START_IDX = 6001
END_IDX = 6050  # or 7000 for full run

for idx in tqdm(range(START_IDX, END_IDX + 1), desc="Running OCR + Translation"):
    img_name = f"tr_img_{idx:05d}"
    img_path = os.path.join(IMAGE_DIR, img_name + ".jpg")

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    # Preprocessing
    gray = to_gray(img)
    clahe_img = apply_clahe(gray, clip)
    denoised = denoise(clahe_img)
    sharpened = sharpen(denoised, alpha)
    upscaled = upscale(sharpened, scale=2)
    preprocessed = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

    # OCR
    result = ocr.ocr(preprocessed, cls=True)

    translations = []
    for line in result:
        for box_info in line:
            box, (text, score) = box_info
            if score < 0.5 or not is_japanese(text):
                continue
            translation = translate_text(text)
            translations.append((text, translation))

    # Save results
    if translations:
        with open(os.path.join(OUTPUT_DIR, f"{img_name}.txt"), "w", encoding="utf-8") as f:
            for jp, en in translations:
                f.write(f"{jp} --> {en}\n")
