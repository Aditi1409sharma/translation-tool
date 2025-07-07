# src/ocr/infer_ocr.py

import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr_model = PaddleOCR(use_angle_cls=True, lang='japan')

# --- Preprocessing ---
def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def apply_clahe(img_gray, clip): return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(img_gray)
def denoise(img_gray): return cv2.fastNlMeansDenoising(img_gray, h=10)
def sharpen(img_gray, alpha): return cv2.filter2D(img_gray, -1, np.array([[0, -1, 0], [-1, 5 * alpha, -1], [0, -1, 0]]))
def upscale(img_gray, scale=2): return cv2.resize(img_gray, (int(img_gray.shape[1]*scale), int(img_gray.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
def dilate(img_gray, kernel_size=1): return cv2.dilate(img_gray, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

def is_japanese(text):
    return any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9faf' for ch in text)

def is_valid_text(text):
    return any(c.isalnum() or '\u4e00' <= c <= '\u9faf' or '\u3040' <= c <= '\u30ff' for c in text)

def sort_boxes(boxes, texts, scores):
    zipped = list(zip(boxes, texts, scores))
    zipped.sort(key=lambda x: (x[0][1], x[0][0]))  # y then x
    return zip(*zipped)

def merge_boxes_by_line(boxes, texts, scores, x_thresh=25, y_thresh=12):
    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]: continue
        x1_min, y1_min, x1_max, y1_max = boxes[i]
        current_texts = [texts[i]]
        current_scores = [scores[i]]
        for j in range(i + 1, len(boxes)):
            if used[j]: continue
            x2_min, y2_min, x2_max, y2_max = boxes[j]
            if abs((y1_min + y1_max)/2 - (y2_min + y2_max)/2) < y_thresh and abs(x2_min - x1_max) < x_thresh:
                used[j] = True
                x1_min = min(x1_min, x2_min)
                y1_min = min(y1_min, y2_min)
                x1_max = max(x1_max, x2_max)
                y1_max = max(y1_max, y2_max)
                current_texts.append(texts[j])
                current_scores.append(scores[j])
        used[i] = True
        merged.append({
            "box": [x1_min, y1_min, x1_max, y1_max],
            "text": "".join(current_texts),
            "score": sum(current_scores)/len(current_scores)
        })
    return merged

def ocr_from_image(image_bgr, clip=1.5, alpha=0.9, scale=2):
    gray = to_gray(image_bgr)
    contrast = apply_clahe(gray, clip)
    denoised = denoise(contrast)
    sharpened = sharpen(denoised, alpha)
    dilated = dilate(sharpened)
    upscaled = upscale(dilated, scale)
    preprocessed = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

    try:
        result = ocr_model.ocr(preprocessed, cls=True)
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        return []

    raw_boxes, raw_texts, raw_scores = [], [], []
    for line in result:
        for box_info in line:
            box, (text, score) = box_info
            if score < 0.25 or not is_japanese(text) or not is_valid_text(text):
                continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            if (max(xs) - min(xs)) * (max(ys) - min(ys)) < 10:
                continue
            raw_boxes.append([min(xs), min(ys), max(xs), max(ys)])
            raw_texts.append(text)
            raw_scores.append(score)

    if not raw_boxes:
        return []

    boxes_sorted, texts_sorted, scores_sorted = sort_boxes(raw_boxes, raw_texts, raw_scores)
    merged = merge_boxes_by_line(list(boxes_sorted), list(texts_sorted), list(scores_sorted))
    return merged
