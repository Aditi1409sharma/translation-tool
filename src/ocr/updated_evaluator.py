import os
import re
import unicodedata
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher

GT_DIR = "C:/Users/aditi/GT"
CSV_DIR = "2"
START_IDX = 6001
END_IDX = 6100

IOU_THRESHOLD = 0.3
SIMILARITY_THRESHOLD = 0.75

def normalize_text(s):
    s = s.lower().strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'\s+', '', s)
    return s

def load_gt_boxes(gt_path):
    boxes = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            lang = parts[8].strip()
            text = parts[9].strip()
            if lang.lower() != "japanese":
                continue
            xs = [float(parts[0]), float(parts[2]), float(parts[4]), float(parts[6])]
            ys = [float(parts[1]), float(parts[3]), float(parts[5]), float(parts[7])]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            norm_text = normalize_text(text)
            boxes.append({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "text": norm_text, "matched": False})
    return boxes

def compute_iou(boxA, boxB):
    xA = max(boxA["xmin"], boxB["xmin"])
    yA = max(boxA["ymin"], boxB["ymin"])
    xB = min(boxA["xmax"], boxB["xmax"])
    yB = min(boxA["ymax"], boxB["ymax"])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA["xmax"] - boxA["xmin"]) * (boxA["ymax"] - boxA["ymin"])
    boxBArea = (boxB["xmax"] - boxB["xmin"]) * (boxB["ymax"] - boxB["ymin"])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def evaluate_csv(csv_file):
    TP, FP, FN = 0, 0, 0

    # Load OCR boxes
    ocr_boxes = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        for l in lines:
            parts = l.strip().split(",")
            if len(parts) < 7:
                continue
            img_name = parts[0]
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            text = normalize_text(parts[5])

            xmin = cx - w/2
            xmax = cx + w/2
            ymin = cy - h/2
            ymax = cy + h/2

            ocr_boxes.setdefault(img_name, []).append({
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, 
                "text": text, "matched": False
            })

    # Process images
    for idx in tqdm(range(START_IDX, END_IDX + 1), desc=f"Evaluating {os.path.basename(csv_file)}"):
        img_name = f"tr_img_{idx:05d}"
        gt_path = os.path.join(GT_DIR, img_name + ".txt")

        if not os.path.exists(gt_path):
            continue

        gt_list = load_gt_boxes(gt_path)
        ocr_list = ocr_boxes.get(img_name, [])

        # Match
        for gt_box in gt_list:
            best_match = None
            best_iou = 0.0

            for ocr_box in ocr_list:
                if ocr_box["matched"]:
                    continue
                iou = compute_iou(gt_box, ocr_box)
                if iou >= IOU_THRESHOLD:
                    sim = similar(gt_box["text"], ocr_box["text"])
                    if sim >= SIMILARITY_THRESHOLD and iou > best_iou:
                        best_match = ocr_box
                        best_iou = iou

            if best_match:
                best_match["matched"] = True
                gt_box["matched"] = True
                TP += 1
            else:
                FN += 1

        FP += sum(1 for ocr_box in ocr_list if not ocr_box["matched"])

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n[{os.path.basename(csv_file)}]")
    print(f"TP = {TP}, FP = {FP}, FN = {FN}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1 Score  = {f1:.4f}\n")

# Main loop
if __name__ == "__main__":
    all_csvs = [os.path.join(CSV_DIR, f) for f in os.listdir(CSV_DIR) if f.startswith("boxes_") and f.endswith(".csv")]

    print(f"Found {len(all_csvs)} CSV files to evaluate.\n")

    for csv_file in all_csvs:
        evaluate_csv(csv_file)























# import os
# import re
# import unicodedata
# import numpy as np
# from tqdm import tqdm
# from difflib import SequenceMatcher

# GT_DIR = "C:/Users/aditi/GT"
# CSV_DIR = "outputs_grid_search"
# START_IDX = 6001
# END_IDX = 6100

# MAX_DISTANCE = 50  # px
# SIMILARITY_THRESHOLD = 0.75

# def normalize_text(s):
#     s = s.lower().strip()
#     s = unicodedata.normalize("NFKC", s)
#     s = re.sub(r'\s+', '', s)
#     return s

# def load_gt_boxes(gt_path):
#     boxes = []
#     with open(gt_path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split(",")
#             if len(parts) < 10:
#                 continue
#             lang = parts[8].strip()
#             text = parts[9].strip()
#             if lang.lower() != "japanese":
#                 continue
#             xs = [float(parts[0]), float(parts[2]), float(parts[4]), float(parts[6])]
#             ys = [float(parts[1]), float(parts[3]), float(parts[5]), float(parts[7])]
#             cx = np.mean(xs)
#             cy = np.mean(ys)
#             norm_text = normalize_text(text)
#             boxes.append({"cx": cx, "cy": cy, "text": norm_text, "matched": False})
#     return boxes

# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# def evaluate_csv(csv_file):
#     TP, FP, FN = 0, 0, 0

#     # Load all OCR boxes in memory
#     ocr_boxes = {}
#     with open(csv_file, "r", encoding="utf-8") as f:
#         lines = f.readlines()[1:]
#         for l in lines:
#             parts = l.strip().split(",")
#             if len(parts) < 7:
#                 continue
#             img_name = parts[0]
#             cx = float(parts[1])
#             cy = float(parts[2])
#             text = normalize_text(parts[5])
#             ocr_boxes.setdefault(img_name, []).append({"cx": cx, "cy": cy, "text": text, "matched": False})

#     # Process images
#     for idx in tqdm(range(START_IDX, END_IDX + 1), desc=f"Evaluating {os.path.basename(csv_file)}"):
#         img_name = f"tr_img_{idx:05d}"
#         gt_path = os.path.join(GT_DIR, img_name + ".txt")

#         if not os.path.exists(gt_path):
#             continue

#         gt_list = load_gt_boxes(gt_path)
#         ocr_list = ocr_boxes.get(img_name, [])

#         # Match
#         for gt_box in gt_list:
#             best_match = None
#             best_dist = float("inf")
#             for ocr_box in ocr_list:
#                 if ocr_box["matched"]:
#                     continue
#                 dist = np.sqrt((gt_box["cx"] - ocr_box["cx"])**2 + (gt_box["cy"] - ocr_box["cy"])**2)
#                 if dist <= MAX_DISTANCE:
#                     sim = similar(gt_box["text"], ocr_box["text"])
#                     if sim >= SIMILARITY_THRESHOLD and dist < best_dist:
#                         best_match = ocr_box
#                         best_dist = dist
#             if best_match:
#                 best_match["matched"] = True
#                 gt_box["matched"] = True
#                 TP += 1
#             else:
#                 FN += 1

#         # Remaining unmatched OCR boxes → FP
#         FP += sum(1 for ocr_box in ocr_list if not ocr_box["matched"])

#     # Metrics
#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

#     print(f"\n[{os.path.basename(csv_file)}]")
#     print(f"TP = {TP}, FP = {FP}, FN = {FN}")
#     print(f"Precision = {precision:.4f}")
#     print(f"Recall    = {recall:.4f}")
#     print(f"F1 Score  = {f1:.4f}\n")

# # Main loop
# if __name__ == "__main__":
#     all_csvs = [os.path.join(CSV_DIR, f) for f in os.listdir(CSV_DIR) if f.startswith("boxes_") and f.endswith(".csv")]

#     print(f"Found {len(all_csvs)} CSV files to evaluate.\n")

#     for csv_file in all_csvs:
#         evaluate_csv(csv_file)
