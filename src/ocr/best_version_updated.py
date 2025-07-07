import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from paddleocr import PaddleOCR

# --- Paths ---
IMAGE_DIR = "C:/Users/aditi/ImagesPart2"
OUTPUT_DIR = "best_version"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_IDX = 6001
END_IDX = 6500

ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# --- Preprocessing Functions ---
def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def apply_clahe(img_gray, clip): return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(img_gray)
def denoise(img_gray): return cv2.fastNlMeansDenoising(img_gray, h=10)
def sharpen(img_gray, alpha): return cv2.filter2D(img_gray, -1, np.array([[0, -1, 0], [-1, 5 * alpha, -1], [0, -1, 0]]))
def upscale(img_gray, scale=2): return cv2.resize(img_gray, (int(img_gray.shape[1]*scale), int(img_gray.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
def dilate(img_gray, kernel_size=1): return cv2.dilate(img_gray, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)

# --- Japanese check ---
def is_japanese(text): return any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9faf' for ch in text)
def is_valid_text(text): return any(c.isalnum() or '\u4e00' <= c <= '\u9faf' or '\u3040' <= c <= '\u30ff' for c in text)

# --- Sort by reading order ---
def sort_boxes(boxes, texts, scores):
    zipped = list(zip(boxes, texts, scores))
    zipped.sort(key=lambda x: (x[0][1], x[0][0]))  # y then x
    return zip(*zipped)

# --- Improved horizontal merge for Japanese line structure ---
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
            # same line & horizontally close
            if abs((y1_min + y1_max) / 2 - (y2_min + y2_max) / 2) < y_thresh and abs(x2_min - x1_max) < x_thresh:
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
            "score": sum(current_scores) / len(current_scores)
        })

    return merged

# --- Main image OCR process ---
def process_image(idx, clip, alpha, scale, csv_file, debug_dir=None):
    img_name = f"tr_img_{idx:05d}"
    img_path = os.path.join(IMAGE_DIR, img_name + ".jpg")
    if not os.path.exists(img_path):
        return 0, 0.0

    img = cv2.imread(img_path)
    gray = to_gray(img)
    contrast = apply_clahe(gray, clip)
    denoised = denoise(contrast)
    sharpened = sharpen(denoised, alpha)
    dilated = dilate(sharpened)
    upscaled = upscale(dilated, scale)
    preprocessed = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

    try:
        result = ocr.ocr(preprocessed, cls=True)
    except Exception as e:
        print(f"[ERROR] OCR failed for {img_name}: {e}")
        return 0, 0.0

    MIN_CONFIDENCE = 0.25
    MIN_BOX_AREA = 10

    raw_boxes, raw_texts, raw_scores = [], [], []

    for line in result:
        for box_info in line:
            box, (text, score) = box_info
            if score < MIN_CONFIDENCE or not is_japanese(text) or not is_valid_text(text):
                continue

            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            if (max(xs) - min(xs)) * (max(ys) - min(ys)) < MIN_BOX_AREA:
                continue

            raw_boxes.append([min(xs), min(ys), max(xs), max(ys)])
            raw_texts.append(text)
            raw_scores.append(score)

    if not raw_boxes:
        return 0, 0.0

    boxes_sorted, texts_sorted, scores_sorted = sort_boxes(raw_boxes, raw_texts, raw_scores)
    merged_results = merge_boxes_by_line(list(boxes_sorted), list(texts_sorted), list(scores_sorted))

    lines = []
    total_score = 0
    vis = img.copy()

    for entry in merged_results:
        x_min, y_min, x_max, y_max = entry["box"]
        text = entry["text"].replace(",", " ").replace('"', "").strip()
        score = entry["score"]

        if not text:
            continue

        cx = (x_min + x_max) / 2 / scale
        cy = (y_min + y_max) / 2 / scale
        w = (x_max - x_min) / scale
        h = (y_max - y_min) / scale

        lines.append(f'{img_name},{cx:.1f},{cy:.1f},{w:.1f},{h:.1f},"{text}",{score:.4f}\n')
        total_score += score

        # Draw box
        if debug_dir:
            x_min, y_min, x_max, y_max = map(lambda v: int(v / scale), [x_min, y_min, x_max, y_max])
            cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(vis, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{img_name}_vis.jpg"), vis)

    with open(csv_file, "a", encoding="utf-8") as f:
        f.writelines(lines)

    return len(merged_results), total_score / len(merged_results)

# --- Grid search runner ---
def main():
    clahe_clip_values = [1.5]
    sharpen_alpha_values = [0.9]
    scale_factors = [2]

    for clip in clahe_clip_values:
        for alpha in sharpen_alpha_values:
            for scale in scale_factors:
                csv_file = os.path.join(OUTPUT_DIR, f"boxes_clip{clip}_alpha{alpha}_scale{scale}.csv")
                debug_dir = os.path.join(OUTPUT_DIR, f"debug_clip{clip}_alpha{alpha}_scale{scale}")

                if os.path.exists(csv_file):
                    print(f"[SKIP] clip={clip}, alpha={alpha}, scale={scale} → already processed.")
                    continue

                with open(csv_file, "w", encoding="utf-8") as f:
                    f.write("Image_Name,Center_X,Center_Y,Width,Height,Text,Score\n")

                print(f"\n>>> Processing: CLAHE={clip}, Sharpen={alpha}, Scale={scale}")

                total_boxes, total_score = 0, 0.0

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        executor.submit(process_image, idx, clip, alpha, scale, csv_file, debug_dir if idx == START_IDX else None)
                        for idx in range(START_IDX, END_IDX + 1)
                    ]
                    for future in tqdm(futures):
                        b, s = future.result()
                        total_boxes += b
                        total_score += s

                avg_score = total_score / total_boxes if total_boxes > 0 else 0.0
                with open(os.path.join(OUTPUT_DIR, f"results_clip{clip}_alpha{alpha}_scale{scale}.csv"), "w", encoding="utf-8") as f:
                    f.write("Parameter,Value\n")
                    f.write(f"CLAHE_clip,{clip}\n")
                    f.write(f"Sharpen_alpha,{alpha}\n")
                    f.write(f"Scale_factor,{scale}\n")
                    f.write(f"Total_boxes,{total_boxes}\n")
                    f.write(f"Avg_confidence,{avg_score:.4f}\n")

if __name__ == "__main__":
    main()
