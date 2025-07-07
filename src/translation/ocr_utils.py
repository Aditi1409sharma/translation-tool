# ocr_utils.py
import cv2
import numpy as np
from paddleocr import PaddleOCR

# --- Fixed Parameters ---
CLAHE_CLIP = 1.5
SHARPEN_ALPHA = 0.9
SCALE_FACTOR = 2
MIN_CONFIDENCE = 0.25
MIN_BOX_AREA = 10

# --- Initialize PaddleOCR (reuse instance) ---
ocr_model = PaddleOCR(use_angle_cls=True, lang='japan')

# --- Preprocessing functions ---
def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_clahe(img_gray, clip=CLAHE_CLIP):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(img_gray)

def denoise(img_gray): return cv2.fastNlMeansDenoising(img_gray, h=10)

def sharpen(img_gray, alpha=SHARPEN_ALPHA):
    kernel = np.array([[0, -1, 0], [-1, 5 * alpha, -1], [0, -1, 0]])
    return cv2.filter2D(img_gray, -1, kernel)

def upscale(img_gray, scale=SCALE_FACTOR):
    return cv2.resize(img_gray, (int(img_gray.shape[1] * scale), int(img_gray.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)

def dilate(img_gray, kernel_size=1):
    return cv2.dilate(img_gray, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

def preprocess_image(img):
    gray = to_gray(img)
    contrast = apply_clahe(gray)
    denoised = denoise(contrast)
    sharpened = sharpen(denoised)
    dilated = dilate(sharpened)
    upscaled = upscale(dilated)
    return cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

# --- Utility ---
def is_japanese(text): return any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9faf' for ch in text)

def is_valid_text(text): return any(c.isalnum() or '\u4e00' <= c <= '\u9faf' or '\u3040' <= c <= '\u30ff' for c in text)

# --- Main OCR Function ---
def perform_ocr(image):
    preprocessed = preprocess_image(image)
    results = ocr_model.ocr(preprocessed, cls=True)

    boxes = []
    texts = []
    for line in results:
        for box_info in line:
            box, (text, score) = box_info
            if score < MIN_CONFIDENCE or not is_japanese(text) or not is_valid_text(text):
                continue

            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            if (max(xs) - min(xs)) * (max(ys) - min(ys)) < MIN_BOX_AREA:
                continue

            boxes.append(((int(xs[0]), int(ys[0])), (int(xs[2]), int(ys[2]))))
            texts.append(text)

    return boxes, texts
