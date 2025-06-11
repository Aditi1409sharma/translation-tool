import cv2
import os
import numpy as np
from tqdm import tqdm
from ocr_infer import preprocess_image, run_final_ocr, ocr 
CONFIDENCE_THRESHOLD = 0.7

IMAGE_DIR = "C:/Users/aditi/ImagesPart2"
LOG_FILE = os.path.join("outputs_steps", "batch_results.csv")


os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


START_IDX = 6001
END_IDX = 7000 


with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("Image,Total_Boxes,Retained_Boxes,Average_Confidence\n")

for idx in tqdm(range(START_IDX, END_IDX + 1), desc="Evaluating Japanese OCR"):
    img_name = f"tr_img_{idx:05d}"
    img_path = os.path.join(IMAGE_DIR, img_name + ".jpg")

    if not os.path.exists(img_path):
        print(f"[WARNING] Missing: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read: {img_path}")
        continue

    preprocessed = preprocess_image(img)

    result = ocr.ocr(preprocessed, cls=True)
    total, retained, score_sum = 0, 0, 0.0

    for line in result:
        for word_info in line:
            total += 1
            _, (text, score) = word_info
            if score >= CONFIDENCE_THRESHOLD:
                retained += 1
                score_sum += score

    avg_conf = score_sum / retained if retained else 0.0

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{img_name},{total},{retained},{avg_conf:.4f}\n")
