import cv2
import os
import numpy as np
from tqdm import tqdm
from ocr_infer import preprocess_image, ocr  # run_final_ocr no longer used here

CONFIDENCE_THRESHOLD = 0.7
IMAGE_DIR = "C:/Users/aditi/ImagesPart2"
LOG_FILE = os.path.join("output_steps", "batch_results.csv")

# --- Ensure Output Directory Exists ---
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

START_IDX = 6001
END_IDX = 7000

# --- Helper to Rotate Image ---
def rotate_image(img, angle):
    if angle == 0:
        return img
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

# --- Choose Best Rotation Based on OCR Confidence ---
def get_best_rotation(img):
    best_score, best_result = -1, None
    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(img, angle)
        result = ocr.ocr(rotated, cls=True)

        score_sum, count = 0, 0
        for line in result:
            for _, (_, score) in line:
                score_sum += score
                count += 1

        avg_score = score_sum / count if count else 0
        if avg_score > best_score:
            best_score = avg_score
            best_result = result

    return best_result

# --- Initialize CSV ---
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("Image,Total_Boxes,Retained_Boxes,Average_Confidence\n")

# --- Batch Evaluation Loop ---
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
    result = get_best_rotation(preprocessed)

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
