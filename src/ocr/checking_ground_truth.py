import cv2
import os
import numpy as np
import pandas as pd
from glob import glob
from itertools import combinations
from paddleocr import PaddleOCR

# --- Individual steps ---

def grayscale(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img): 
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21) if len(img.shape) == 2 else cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def deskew(img):
    gray = grayscale(img) if len(img.shape) == 3 else img
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    if coords.shape[0] < 2: 
        return img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def binarize_otsu(img):
    gray = grayscale(img) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def binarize_adaptive(img):
    gray = grayscale(img) if len(img.shape) == 3 else img
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 15, 11)

# --- Mapping
step_functions = {
    "grayscale": grayscale,
    "denoise": denoise,
    "deskew": deskew,
    "binarize_otsu": binarize_otsu,
    "binarize_adaptive": binarize_adaptive
}

# --- Apply step sequence to image
def apply_preprocessing_pipeline(img, steps):
    for step in steps:
        img = step_functions[step](img)
    return img

# --- IOU function for box matching
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# --- Compare ground truth vs OCR boxes, returns F1 score
def gt_vs_ocr(gt_path, ocr_boxes, iou_thresh=0.5):
    try:
        import json
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_boxes = json.load(f)  # Expected format: list of [x1,y1,x2,y2]
    except Exception as e:
        print(f"Could not load ground truth file {gt_path}: {e}")
        return 0.0

    matched_gt = set()
    matched_ocr = set()

    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(ocr_boxes):
            if i in matched_gt or j in matched_ocr:
                continue
            if iou(gt, pred) >= iou_thresh:
                matched_gt.add(i)
                matched_ocr.add(j)

    tp = len(matched_gt)
    fp = len(ocr_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return f1

# --- PaddleOCR initialization (det and rec enabled)
ocr = PaddleOCR(use_angle_cls=True, lang='japan', show_log=False)  # 'japan' model for Japanese text, adjust as needed

# --- Run OCR on image and extract bounding boxes
def run_ocr(img):
    # PaddleOCR expects RGB images
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = ocr.ocr(img_rgb, cls=True)
    boxes = []
    for line in result:
        # line[0] is box coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        pts = line[0]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, y_min = int(min(xs)), int(min(ys))
        x_max, y_max = int(max(xs)), int(max(ys))
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes

# --- Main evaluation ---

image_dir = './images'  # Change paths accordingly
gt_dir = './gt'

image_paths = sorted(glob(os.path.join(image_dir, '*.png')))
all_steps = list(step_functions.keys())

results = []

# Generate all combinations (length 1 to len)
for r in range(1, len(all_steps)+1):
    for steps in combinations(all_steps, r):
        if "grayscale" not in steps:
            continue
        steps = list(steps)
        print(f"\nTrying combo: {steps}")
        scores = []

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            gt_path = os.path.join(gt_dir, filename.replace('.png', '.json'))

            img = cv2.imread(img_path)
            try:
                processed = apply_preprocessing_pipeline(img, steps)
                boxes = run_ocr(processed)
                f1 = gt_vs_ocr(gt_path, boxes)
                scores.append(f1)
            except Exception as e:
                print(f"Error processing {filename} with {steps}: {e}")
                continue

        if scores:
            avg_f1 = np.mean(scores)
            results.append((" → ".join(steps), avg_f1))
            print(f"Combo {steps} → F1: {avg_f1:.4f}")

# Save final results
df = pd.DataFrame(results, columns=["Preprocessing_Pipeline", "Avg_F1_Score"])
df.sort_values("Avg_F1_Score", ascending=False, inplace=True)
df.to_csv("combo_results.csv", index=False)
print("\nTop combinations:")
print(df.head())
