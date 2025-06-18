import cv2
import os
import numpy as np
from tqdm import tqdm
from ocr_infer import preprocess_image, ocr

CONFIDENCE_THRESHOLD = 0.7
IMAGE_DIR = "C:/Users/aditi/ImagesPart2"
GT_DIR = "C:/Users/aditi/GT"
LOG_FILE = os.path.join("outputs_steps", "batch_results_filtered.csv")

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
START_IDX = 6001
END_IDX = 7000

# --- Preprocessing Functions ---
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_clahe(img_gray, clip=3.0):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(img_gray)

def denoise(img_gray):
    return cv2.fastNlMeansDenoising(img_gray, h=10)

def sharpen(img_gray, alpha=1.5):
    kernel = np.array([[0, -1, 0],
                       [-1, 5 * alpha, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img_gray, -1, kernel)

def upscale(img_gray, scale=2):
    h, w = img_gray.shape
    return cv2.resize(img_gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

# --- Initialize CSV ---
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("Image,Total_Boxes,Retained_Boxes,Avg_Conf,Japanese_Text\n")

# --- Batch OCR Loop ---
for idx in tqdm(range(START_IDX, END_IDX + 1), desc="OCR with filtering"):
    img_name = f"tr_img_{idx:05d}"
    img_path = os.path.join(IMAGE_DIR, img_name + ".jpg")
    gt_path = os.path.join(GT_DIR, img_name + ".txt")

    if not os.path.exists(img_path):
        print(f"[WARNING] Missing: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read: {img_path}")
        continue

    # --- Preprocessing ---
    gray = to_gray(img)
    contrast = apply_clahe(gray, clip=3.0)
    denoised = denoise(contrast)
    sharpened = sharpen(denoised, alpha=1.2)
    upscaled = upscale(sharpened, scale=2)

    preprocessed = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

    # --- OCR ---
    result = ocr.ocr(preprocessed, cls=True)

    # --- Japanese filter ---
    def is_japanese(text):
        return any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9faf' for ch in text)

    japanese_texts = []
    total, retained, score_sum = 0, 0, 0.0

    for line in result:
        for word_info in line:
            total += 1
            _, (text, score) = word_info
            if score >= CONFIDENCE_THRESHOLD:
                retained += 1
                score_sum += score
                if is_japanese(text):
                    japanese_texts.append(text)

    avg_conf = score_sum / retained if retained else 0.0
    jp_text_str = " | ".join(japanese_texts)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{img_name},{total},{retained},{avg_conf:.4f},\"{jp_text_str}\"\n")
