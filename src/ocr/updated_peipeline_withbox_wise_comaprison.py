import cv2
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from paddleocr import PaddleOCR
from difflib import SequenceMatcher

# --- Paths ---
IMAGE_DIR = "C:/Users/aditi/ImagesPart2"
OUTPUT_DIR = "2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
START_IDX = 6001
END_IDX = 6200

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


# --- Japanese text filter ---
def is_japanese(text):
    return any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9faf' for ch in text)

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
    upscaled = upscale(sharpened, scale=scale)
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

    # Run OCR
    try:
        result = ocr.ocr(preprocessed, cls=True)
    except Exception as e:
        print(f"[ERROR] OCR failed for {img_name}: {e}")
        return 0, 0

    # Filter and write results
    MIN_CONFIDENCE = 0.4
    MIN_BOX_AREA = 50
    scale_factor = scale

    lines_to_write = []
    total_boxes = 0
    total_conf = 0.0

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

            center_x = np.mean(xs)
            center_y = np.mean(ys)
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)

            # Scale back to original image coords
            center_x_orig = center_x / scale_factor
            center_y_orig = center_y / scale_factor
            width_orig = width / scale_factor
            height_orig = height / scale_factor

            text_clean = text.replace(",", " ").replace('"', "")
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
    clahe_clip_values = [1.7, 2.1]
    sharpen_alpha_values = [1.0]
    scale_factors = [2.5]

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
# from tqdm import tqdm
# from ocr_infer import preprocess_image, ocr
# from difflib import SequenceMatcher
# from concurrent.futures import ProcessPoolExecutor


# IMAGE_DIR = "C:/Users/aditi/ImagesPart2"
# GT_DIR = "C:/Users/aditi/GT"
# OUTPUT_DIR = "outputs_parallel_grid_search"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# START_IDX = 6001
# END_IDX = 6100

# # Preprocessing Functions
# def to_gray(img):
#     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# def apply_clahe(img_gray, clip):
#     clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
#     return clahe.apply(img_gray)

# def denoise(img_gray):
#     return cv2.fastNlMeansDenoising(img_gray, h=10)

# def sharpen(img_gray, alpha):
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5 * alpha, -1],
#                        [0, -1, 0]])
#     return cv2.filter2D(img_gray, -1, kernel)

# def upscale(img_gray, scale=2):
#     h, w = img_gray.shape
#     return cv2.resize(img_gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

# # Japanese filter
# def is_japanese(text):
#     return any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9faf' for ch in text)

# # Fuzzy match
# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# # Grid Search Params
# clahe_clip_values = [2.0, 1.8, 1.9]
# sharpen_alpha_values = [0.7, 0.9, 0.8]

# # Thresholds
# MIN_CONFIDENCE = 0.5
# MIN_BOX_AREA = 100
# scale_factor = 2

# # Parallel params
# NUM_THREADS = 8  # Adjust based on your CPU

# # Process one image
# def process_image(idx, clip, alpha, csv_file, debug_dir):
#     img_name = f"tr_img_{idx:05d}"
#     img_path = os.path.join(IMAGE_DIR, img_name + ".jpg")

#     if not os.path.exists(img_path):
#         return

#     img = cv2.imread(img_path)
#     if img is None:
#         return

#     # Preprocessing
#     gray = to_gray(img)
#     contrast = apply_clahe(gray, clip=clip)
#     denoised = denoise(contrast)
#     sharpened = sharpen(denoised, alpha=alpha)
#     upscaled = upscale(sharpened, scale=scale_factor)
#     preprocessed = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

#     # Save intermediate for first image
#     if idx == START_IDX:
#         os.makedirs(debug_dir, exist_ok=True)
#         cv2.imwrite(os.path.join(debug_dir, f"{img_name}_step1_gray.jpg"), gray)
#         cv2.imwrite(os.path.join(debug_dir, f"{img_name}_step2_clahe.jpg"), contrast)
#         cv2.imwrite(os.path.join(debug_dir, f"{img_name}_step3_denoise.jpg"), denoised)
#         cv2.imwrite(os.path.join(debug_dir, f"{img_name}_step4_sharpen.jpg"), sharpened)
#         cv2.imwrite(os.path.join(debug_dir, f"{img_name}_step5_upscaled.jpg"), upscaled)
#         cv2.imwrite(os.path.join(debug_dir, f"{img_name}_final_preprocessed.jpg"), preprocessed)

#     # OCR
#     result = ocr.ocr(preprocessed, cls=True)

#     # Write CSV
#     lines_to_write = []
#     for line in result:
#         for box_info in line:
#             box, (text, score) = box_info

#             if score < MIN_CONFIDENCE:
#                 continue

#             xs = [pt[0] for pt in box]
#             ys = [pt[1] for pt in box]
#             box_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
#             if box_area < MIN_BOX_AREA:
#                 continue

#             center_x = np.mean(xs)
#             center_y = np.mean(ys)
#             width = max(xs) - min(xs)
#             height = max(ys) - min(ys)

#             center_x_orig = center_x / scale_factor
#             center_y_orig = center_y / scale_factor
#             width_orig = width / scale_factor
#             height_orig = height / scale_factor

#             if is_japanese(text):
#                 text_clean = text.replace(",", " ").replace('"', "")
#                 lines_to_write.append(
#                     f"{img_name},{center_x_orig:.1f},{center_y_orig:.1f},{width_orig:.1f},{height_orig:.1f},\"{text_clean}\",{score:.4f}\n"
#                 )

#     if lines_to_write:
#         with open(csv_file, "a", encoding="utf-8") as f:
#             f.writelines(lines_to_write)

# # Main loop
# # for clip in clahe_clip_values:
# #     for alpha in sharpen_alpha_values:

# #         log_file = os.path.join(OUTPUT_DIR, f"results_clip{clip}_alpha{alpha}.csv")
# #         csv_file = os.path.join(OUTPUT_DIR, f"boxes_clip{clip}_alpha{alpha}.csv")
# #         debug_dir = os.path.join(OUTPUT_DIR, f"debug_clip{clip}_alpha{alpha}")

# #         if os.path.exists(log_file):
# #             print(f"[SKIP] clip={clip}, alpha={alpha} --> already exists.")
# #             continue

# #         print(f"\n>>> Processing: CLAHE clip={clip}, Sharpen alpha={alpha}")

# #         # Init result CSVs
# #         with open(log_file, "w", encoding="utf-8") as f:
# #             f.write("Image,Total_Boxes,Avg_Conf,Japanese_Texts\n")

# #         with open(csv_file, "w", encoding="utf-8") as f:
# #             f.write("Image_Name,Center_X,Center_Y,Width,Height,Text,Score\n")

# #         # Parallel execution
# #         with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
# #             list(tqdm(
# #                 executor.map(lambda idx: process_image(idx, clip, alpha, csv_file, debug_dir),
# #                              range(START_IDX, END_IDX + 1)),
# #                 total=(END_IDX - START_IDX + 1),
# #                 desc=f"clip={clip}, alpha={alpha} (Parallel)"
# #             ))




# if __name__ == "__main__":
#     from functools import partial
#     from concurrent.futures import ProcessPoolExecutor

#     # for each param combo
#     for clip in clahe_clip_values:
#         for alpha in sharpen_alpha_values:

#             log_file = os.path.join(OUTPUT_DIR, f"results_clip{clip}_alpha{alpha}.csv")
#             csv_file = os.path.join(OUTPUT_DIR, f"boxes_clip{clip}_alpha{alpha}.csv")

#             if os.path.exists(log_file):
#                 print(f"[SKIP] clip={clip}, alpha={alpha} --> already exists.")
#                 continue

#             with open(log_file, "w", encoding="utf-8") as f:
#                 f.write("Image,Total_Boxes,Avg_Conf,Japanese_Texts\n")

#             print(f"\n>>> Processing: CLAHE clip={clip}, Sharpen alpha={alpha}")

#             debug_dir = os.path.join(OUTPUT_DIR, f"debug_clip{clip}_alpha{alpha}")
#             os.makedirs(debug_dir, exist_ok=True)

#             process_fn = partial(process_image, clip=clip, alpha=alpha, csv_file=csv_file, debug_dir=debug_dir)

#             with ProcessPoolExecutor(max_workers=4) as executor:
#                 list(tqdm(
#                     executor.map(process_fn, range(START_IDX, END_IDX + 1)),
#                     total=END_IDX - START_IDX + 1
#                 ))




















# import cv2
# import os
# import numpy as np
# from tqdm import tqdm
# from ocr_infer import preprocess_image, ocr
# from difflib import SequenceMatcher

# IMAGE_DIR = "C:/Users/aditi/ImagesPart2"
# GT_DIR = "C:/Users/aditi/GT"
# OUTPUT_DIR = "outputs_grid_search"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# START_IDX = 6001
# END_IDX = 6100 

# # --- Preprocessing Functions ---
# def to_gray(img):
#     return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# def apply_clahe(img_gray, clip):
#     clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
#     return clahe.apply(img_gray)

# def denoise(img_gray):
#     return cv2.fastNlMeansDenoising(img_gray, h=10)

# def sharpen(img_gray, alpha):
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5 * alpha, -1],
#                        [0, -1, 0]])
#     return cv2.filter2D(img_gray, -1, kernel)

# def upscale(img_gray, scale=2):
#     h, w = img_gray.shape
#     return cv2.resize(img_gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

# # --- Japanese filter ---
# def is_japanese(text):
#     return any('\u3040' <= ch <= '\u30ff' or '\u4e00' <= ch <= '\u9faf' for ch in text)

# # --- Fuzzy match ---
# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# # --- Example dictionary filter (optional) ---
# japanese_dictionary = {"ありがとう", "こんにちは", "さようなら", "おはよう", "こんばんは"}

# def is_in_dictionary(text):
#     return any(word in text for word in japanese_dictionary)

# # --- Grid Search Params ---
# clahe_clip_values = [2.0, 1.8, 1.6]
# sharpen_alpha_values = [0.7, 0.9, 1.1 ]

# # Thresholds
# MIN_CONFIDENCE = 0.5
# MIN_BOX_AREA = 100

# # --- Upscale factor ---
# scale_factor = 2  # you upscale the image by factor 2

# for clip in clahe_clip_values:
#     for alpha in sharpen_alpha_values:

#         log_file = os.path.join(OUTPUT_DIR, f"results_clip{clip}_alpha{alpha}.csv")

#         if os.path.exists(log_file):
#             print(f"[SKIP] clip={clip}, alpha={alpha} --> already exists.")
#             continue

#         with open(log_file, "w", encoding="utf-8") as f:
#             f.write("Image,Total_Boxes,Avg_Conf,Japanese_Texts\n")

#         print(f"\n>>> Processing: CLAHE clip={clip}, Sharpen alpha={alpha}")

#         for idx in tqdm(range(START_IDX, END_IDX + 1), desc=f"clip={clip}, alpha={alpha}"):
#             img_name = f"tr_img_{idx:05d}"
#             img_path = os.path.join(IMAGE_DIR, img_name + ".jpg")

#             if not os.path.exists(img_path):
#                 print(f"[WARNING] Missing: {img_path}")
#                 continue

#             img = cv2.imread(img_path)
#             if img is None:
#                 print(f"[ERROR] Could not read: {img_path}")
#                 continue

#             # --- Preprocessing ---
#             gray = to_gray(img)
#             contrast = apply_clahe(gray, clip=clip)
#             denoised = denoise(contrast)
#             sharpened = sharpen(denoised, alpha=alpha)
#             upscaled = upscale(sharpened, scale=scale_factor)
#             preprocessed = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

#             # --- Save intermediate steps for FIRST image ---
#             if idx == START_IDX:
#                 debug_dir = os.path.join(OUTPUT_DIR, f"debug_clip{clip}_alpha{alpha}")
#                 os.makedirs(debug_dir, exist_ok=True)

#                 cv2.imwrite(os.path.join(debug_dir, f"{img_name}_step1_gray.jpg"), gray)
#                 cv2.imwrite(os.path.join(debug_dir, f"{img_name}_step2_clahe.jpg"), contrast)
#                 cv2.imwrite(os.path.join(debug_dir, f"{img_name}_step3_denoise.jpg"), denoised)
#                 cv2.imwrite(os.path.join(debug_dir, f"{img_name}_step4_sharpen.jpg"), sharpened)
#                 cv2.imwrite(os.path.join(debug_dir, f"{img_name}_step5_upscaled.jpg"), upscaled)
#                 cv2.imwrite(os.path.join(debug_dir, f"{img_name}_final_preprocessed.jpg"), preprocessed)

#             # --- OCR ---
#             result = ocr.ocr(preprocessed, cls=True)

#             csv_file = os.path.join(OUTPUT_DIR, f"boxes_clip{clip}_alpha{alpha}.csv")

#             # Init CSV
#             if not os.path.exists(csv_file):
#                 with open(csv_file, "w", encoding="utf-8") as f:
#                     f.write("Image_Name,Center_X,Center_Y,Width,Height,Text,Score\n")

#             # Append boxes
#             with open(csv_file, "a", encoding="utf-8") as f:

#                 for line in result:
#                     for box_info in line:
#                         box, (text, score) = box_info

#                         # Filter
#                         if score < MIN_CONFIDENCE:
#                             continue

#                         xs = [pt[0] for pt in box]
#                         ys = [pt[1] for pt in box]
#                         box_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
#                         if box_area < MIN_BOX_AREA:
#                             continue

#                         center_x = np.mean(xs)
#                         center_y = np.mean(ys)
#                         width = max(xs) - min(xs)
#                         height = max(ys) - min(ys)

#                         # 💥 Scale back coordinates to original image space 💥
#                         center_x_orig = center_x / scale_factor
#                         center_y_orig = center_y / scale_factor
#                         width_orig = width / scale_factor
#                         height_orig = height / scale_factor

#                         if is_japanese(text):
#                             text_clean = text.replace(",", " ").replace('"', "")
#                             f.write(f"{img_name},{center_x_orig:.1f},{center_y_orig:.1f},{width_orig:.1f},{height_orig:.1f},\"{text_clean}\",{score:.4f}\n")
