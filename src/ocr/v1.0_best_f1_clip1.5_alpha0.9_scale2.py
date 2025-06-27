import cv2
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from paddleocr import PaddleOCR
from difflib import SequenceMatcher

# --- Paths ---
IMAGE_DIR = "C:/Users/aditi/ImagesPart2"
OUTPUT_DIR = "3_lower_thresh_2_500"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
START_IDX = 6001
END_IDX = 6500

# Initialize OCR once, thread-safe for PaddleOCR (shared read-only)
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# --- Preprocessing Functions ---
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_clahe(img_gray, clip):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(img_gray)

def denoise(img_gray):
    return cv2.fastNlMeansDenoising(img_gray, h=10)

def sharpen(img_gray, alpha):
    kernel = np.array([[0, -1, 0],
                       [-1, 5 * alpha, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img_gray, -1, kernel)

def upscale(img_gray, scale=2):
    h, w = img_gray.shape
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def dilate(img_gray, kernel_size=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img_gray, kernel, iterations=1)


# --- Japanese text filter ---
def is_japanese(text):
    return any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9faf' for ch in text)

def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def merge_boxes(boxes, texts, scores, iou_thresh=0.5):
    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue
        x1_min, y1_min, x1_max, y1_max = boxes[i]
        current_texts = [texts[i]]
        current_scores = [scores[i]]

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            x2_min, y2_min, x2_max, y2_max = boxes[j]
            if iou([x1_min, y1_min, x1_max, y1_max], [x2_min, y2_min, x2_max, y2_max]) > iou_thresh:
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
            "text": " ".join(current_texts),
            "score": sum(current_scores) / len(current_scores)
        })

    return merged


# --- OCR process for one image ---
def process_image(idx, clip, alpha, scale, csv_file, debug_dir=None):
    img_name = f"tr_img_{idx:05d}"
    img_path = os.path.join(IMAGE_DIR, img_name + ".jpg")
    if not os.path.exists(img_path):
        print(f"[WARNING] Missing: {img_path}")
        return 0, 0  # no boxes, no OCR done

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read: {img_path}")
        return 0, 0

    # Preprocessing pipeline
    gray = to_gray(img)
    contrast = apply_clahe(gray, clip=clip)
    denoised = denoise(contrast)
    sharpened = sharpen(denoised, alpha=alpha)
    dilated = dilate(sharpened, kernel_size=1)
    upscaled = upscale(dilated, scale=scale)

    preprocessed = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

    # Save debug images for first image only
    if debug_dir and idx == START_IDX:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{img_name}_gray.jpg"), gray)
        cv2.imwrite(os.path.join(debug_dir, f"{img_name}_clahe.jpg"), contrast)
        cv2.imwrite(os.path.join(debug_dir, f"{img_name}_denoise.jpg"), denoised)
        cv2.imwrite(os.path.join(debug_dir, f"{img_name}_sharpen.jpg"), sharpened)
        cv2.imwrite(os.path.join(debug_dir, f"{img_name}_upscaled.jpg"), upscaled)
        cv2.imwrite(os.path.join(debug_dir, f"{img_name}_final_preprocessed.jpg"), preprocessed)
        cv2.imwrite(os.path.join(debug_dir, f"{img_name}_dilated.jpg"), dilated)


    # Run OCR
    try:
        result = ocr.ocr(preprocessed, cls=True)
    except Exception as e:
        print(f"[ERROR] OCR failed for {img_name}: {e}")
        return 0, 0

    # Filter and write results
    MIN_CONFIDENCE = 0.25
    MIN_BOX_AREA = 10
    IOU_THRESHOLD = 0.5
    scale_factor = scale

    lines_to_write = []
    total_boxes = 0
    total_conf = 0.0

    all_boxes = []
    all_texts = []
    all_scores = []

    for line in result:
        for box_info in line:
            box, (text, score) = box_info
            if score < MIN_CONFIDENCE:
                continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            box_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if box_area < MIN_BOX_AREA:
                continue
            if not is_japanese(text):
                continue

            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)

            all_boxes.append([x_min, y_min, x_max, y_max])
            all_texts.append(text)
            all_scores.append(score)

    merged_results = merge_boxes(all_boxes, all_texts, all_scores, iou_thresh=IOU_THRESHOLD)

    for entry in merged_results:
        x_min, y_min, x_max, y_max = entry["box"]
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        # Scale back
        center_x_orig = center_x / scale_factor
        center_y_orig = center_y / scale_factor
        width_orig = width / scale_factor
        height_orig = height / scale_factor

        text_clean = entry["text"].replace(",", " ").replace('"', "")
        score = entry["score"]

        line_str = f'{img_name},{center_x_orig:.1f},{center_y_orig:.1f},{width_orig:.1f},{height_orig:.1f},"{text_clean}",{score:.4f}\n'
        lines_to_write.append(line_str)
        total_boxes += 1
        total_conf += score


    # Append lines to CSV (thread-safe by writing inside thread but with file lock or sequential writes)
    with open(csv_file, "a", encoding="utf-8") as f:
        f.writelines(lines_to_write)

    avg_conf = (total_conf / total_boxes) if total_boxes > 0 else 0.0
    return total_boxes, avg_conf

# --- Main parameter grid search ---
def main():
    clahe_clip_values = clahe_clip_values = [1.5]

    sharpen_alpha_values = [0.9]
    scale_factors = [2]

    for clip in clahe_clip_values:
        for alpha in sharpen_alpha_values:
            for scale in scale_factors:
                csv_file = os.path.join(OUTPUT_DIR, f"boxes_clip{clip}_alpha{alpha}_scale{scale}.csv")
                debug_dir = os.path.join(OUTPUT_DIR, f"debug_clip{clip}_alpha{alpha}_scale{scale}")

                if os.path.exists(csv_file):
                    print(f"[SKIP] clip={clip}, alpha={alpha}, scale={scale} --> output exists.")
                    continue

                # Init CSV with header
                with open(csv_file, "w", encoding="utf-8") as f:
                    f.write("Image_Name,Center_X,Center_Y,Width,Height,Text,Score\n")

                print(f"\n>>> Processing: CLAHE clip={clip}, Sharpen alpha={alpha}, Scale={scale}")

                total_boxes_all = 0
                total_conf_all = 0.0

                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    for idx in range(START_IDX, END_IDX + 1):
                        futures.append(executor.submit(process_image, idx, clip, alpha, scale, csv_file, debug_dir))

                    for future in tqdm(futures, total=len(futures), desc=f"clip={clip}, alpha={alpha}, scale={scale}"):
                        boxes, conf = future.result()
                        total_boxes_all += boxes
                        total_conf_all += conf

                avg_conf_overall = (total_conf_all / total_boxes_all) if total_boxes_all > 0 else 0.0

                # Optionally save summary log
                log_file = os.path.join(OUTPUT_DIR, f"results_clip{clip}_alpha{alpha}_scale{scale}.csv")
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("Parameter,Value\n")
                    f.write(f"CLAHE_clip,{clip}\n")
                    f.write(f"Sharpen_alpha,{alpha}\n")
                    f.write(f"Scale_factor,{scale}\n")
                    f.write(f"Total_boxes,{total_boxes_all}\n")
                    f.write(f"Avg_confidence,{avg_conf_overall:.4f}\n")

if __name__ == "__main__":
    main()









# import cv2
# import os
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor
# from paddleocr import PaddleOCR

# # --- Paths ---
# IMAGE_DIR = "C:/Users/aditi/ImagesPart2"
# OUTPUT_DIR = "3_lower_thresh_2"
# BAD_IMAGE_LIST = "low_score_translation_samples.csv"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Load bad image list
# bad_df = pd.read_csv(BAD_IMAGE_LIST)
# bad_image_ids = bad_df["Image"].apply(lambda x: x.replace(".txt", "")).tolist()

# # Initialize OCR
# ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# # --- Preprocessing ---
# def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# def apply_clahe(img_gray, clip): return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(img_gray)
# def denoise(img_gray): return cv2.fastNlMeansDenoising(img_gray, h=10)
# def sharpen(img_gray, alpha): return cv2.filter2D(img_gray, -1, np.array([[0, -1, 0], [-1, 5 * alpha, -1], [0, -1, 0]]))
# def upscale(img_gray, scale=2): return cv2.resize(img_gray, (int(img_gray.shape[1]*scale), int(img_gray.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
# def dilate(img_gray, kernel_size=1): return cv2.dilate(img_gray, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
# def is_japanese(text): return any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9faf' for ch in text)

# # --- IOU + Merge boxes ---
# def iou(box1, box2):
#     xi1, yi1 = max(box1[0], box2[0]), max(box1[1], box2[1])
#     xi2, yi2 = min(box1[2], box2[2]), min(box1[3], box2[3])
#     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#     area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
#     area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
#     union_area = area1 + area2 - inter_area
#     return inter_area / union_area if union_area > 0 else 0

# def merge_boxes(boxes, texts, scores, iou_thresh=0.5):
#     merged, used = [], [False]*len(boxes)
#     for i in range(len(boxes)):
#         if used[i]: continue
#         b = boxes[i]
#         curr_texts, curr_scores = [texts[i]], [scores[i]]
#         for j in range(i+1, len(boxes)):
#             if used[j]: continue
#             if iou(b, boxes[j]) > iou_thresh:
#                 used[j] = True
#                 b = [min(b[0], boxes[j][0]), min(b[1], boxes[j][1]),
#                      max(b[2], boxes[j][2]), max(b[3], boxes[j][3])]
#                 curr_texts.append(texts[j])
#                 curr_scores.append(scores[j])
#         used[i] = True
#         merged.append({
#             "box": b,
#             "text": " ".join(curr_texts).strip(),  # Strip whitespace
#             "score": np.mean(curr_scores)
#         })
#     return merged

# # --- OCR + Save boxes and visual ---
# def process_image(img_name, clip, alpha, scale, csv_file, debug_dir=None):
#     img_path = os.path.join(IMAGE_DIR, img_name + ".jpg")
#     if not os.path.exists(img_path):
#         print(f"[⚠️] Image missing: {img_path}")
#         return 0, 0

#     img = cv2.imread(img_path)
#     gray = to_gray(img)
#     contrast = apply_clahe(gray, clip)
#     denoised = denoise(contrast)
#     sharpened = sharpen(denoised, alpha)
#     dilated = dilate(sharpened, 1)
#     upscaled = upscale(dilated, scale)
#     preprocessed = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

#     try:
#         result = ocr.ocr(preprocessed, cls=True)
#     except Exception as e:
#         print(f"[OCR FAIL] {img_name}: {e}")
#         return 0, 0

#     all_boxes, all_texts, all_scores = [], [], []
#     for line in result:
#         for box_info in line:
#             box, (text, score) = box_info
#             if score < 0.25 or not is_japanese(text):
#                 continue
#             xs, ys = [pt[0] for pt in box], [pt[1] for pt in box]
#             if (max(xs)-min(xs)) * (max(ys)-min(ys)) < 10:
#                 continue
#             all_boxes.append([min(xs), min(ys), max(xs), max(ys)])
#             all_texts.append(text)
#             all_scores.append(score)

#     merged = merge_boxes(all_boxes, all_texts, all_scores)
#     lines, final_merged = [], []
#     for entry in merged:
#         text_clean = entry["text"].replace(",", " ").replace('"', "").strip()
#         if not text_clean:  # Skip if empty after cleaning
#             continue

#         x_min, y_min, x_max, y_max = entry["box"]
#         cx, cy = (x_min + x_max) / 2 / scale, (y_min + y_max) / 2 / scale
#         w, h = (x_max - x_min) / scale, (y_max - y_min) / scale

#         lines.append(f'{img_name},{cx:.1f},{cy:.1f},{w:.1f},{h:.1f},"{text_clean}",{entry["score"]:.4f}\n')
#         final_merged.append((entry["box"], text_clean))

#     with open(csv_file, "a", encoding="utf-8") as f:
#         f.writelines(lines)

#     # Save visual only for boxes with valid text
#     # Save visual only for boxes with valid text
#     vis = img.copy()
#     for box, label in final_merged:
#         if not label.strip():
#             continue
#         # Scale back box coordinates
#         x_min, y_min, x_max, y_max = [int(coord / scale) for coord in box]
#         cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
#         cv2.putText(vis, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
#     cv2.imwrite(os.path.join(OUTPUT_DIR, f"{img_name}_ocr_vis.jpg"), vis)


#     return len(final_merged), np.mean([entry["score"] for entry in merged]) if final_merged else 0.0

# # --- Run on bad images only ---
# def main():
#     clip, alpha, scale = 1.5, 0.9, 2
#     csv_file = os.path.join(OUTPUT_DIR, f"bad_boxes_clip{clip}_alpha{alpha}_scale{scale}.csv")

#     with open(csv_file, "w", encoding="utf-8") as f:
#         f.write("Image_Name,Center_X,Center_Y,Width,Height,Text,Score\n")

#     total_boxes_all, total_conf_all = 0, 0.0
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         futures = []
#         for img_name in bad_image_ids:
#             futures.append(executor.submit(process_image, img_name, clip, alpha, scale, csv_file))
#         for future in tqdm(futures, total=len(futures), desc="Processing bad images"):
#             boxes, conf = future.result()
#             total_boxes_all += boxes
#             total_conf_all += conf

#     avg_conf = total_conf_all / total_boxes_all if total_boxes_all > 0 else 0
#     print(f"\n✅ Finished: {total_boxes_all} boxes with avg confidence {avg_conf:.4f}")

# if __name__ == "__main__":
#     main()
