import os
import re
import unicodedata
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher

# Your directories
GT_DIR = "C:/Users/aditi/GT"
CSV_DIR = "3_lower_thresh_2"  # use your OUTPUT_DIR here
START_IDX = 6001
END_IDX = 6100

MAX_DISTANCE = 50  # px
SIMILARITY_THRESHOLD = 0.5

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
            cx = np.mean(xs)
            cy = np.mean(ys)
            norm_text = normalize_text(text)
            boxes.append({"cx": cx, "cy": cy, "text": norm_text, "matched": False})
    return boxes

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def evaluate_csv(csv_file):
    TP, FP, FN = 0, 0, 0

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
            text = normalize_text(parts[5])
            ocr_boxes.setdefault(img_name, []).append({"cx": cx, "cy": cy, "text": text, "matched": False})

    for idx in range(START_IDX, END_IDX + 1):
        img_name = f"tr_img_{idx:05d}"
        gt_path = os.path.join(GT_DIR, img_name + ".txt")

        if not os.path.exists(gt_path):
            continue

        gt_list = load_gt_boxes(gt_path)
        ocr_list = ocr_boxes.get(img_name, [])

        for gt_box in gt_list:
            best_match = None
            best_dist = float("inf")
            for ocr_box in ocr_list:
                if ocr_box["matched"]:
                    continue
                dist = np.sqrt((gt_box["cx"] - ocr_box["cx"])**2 + (gt_box["cy"] - ocr_box["cy"])**2)
                if dist <= MAX_DISTANCE:
                    sim = similar(gt_box["text"], ocr_box["text"])
                    if sim >= SIMILARITY_THRESHOLD and dist < best_dist:
                        best_match = ocr_box
                        best_dist = dist
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

    return TP, FP, FN, precision, recall, f1

# --- MAIN ---
if __name__ == "__main__":
    all_csvs = [os.path.join(CSV_DIR, f) for f in os.listdir(CSV_DIR) if f.startswith("boxes_") and f.endswith(".csv")]
    print(f"Found {len(all_csvs)} CSV files to evaluate.\n")

    results = []

    for csv_file in tqdm(all_csvs, desc="Evaluating"):
        TP, FP, FN, precision, recall, f1 = evaluate_csv(csv_file)
        clip_search = re.search(r'clip([\d\.]+)_', csv_file)
        alpha_search = re.search(r'alpha([\d\.]+)\.csv', csv_file)
        clip = float(clip_search.group(1)) if clip_search else -1
        alpha = float(alpha_search.group(1)) if alpha_search else -1

        print(f"\n[{os.path.basename(csv_file)}]")
        print(f"TP = {TP}, FP = {FP}, FN = {FN}")
        print(f"Precision = {precision:.4f}")
        print(f"Recall    = {recall:.4f}")
        print(f"F1 Score  = {f1:.4f}")

        results.append({
            "csv": os.path.basename(csv_file),
            "clip": clip,
            "alpha": alpha,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    # Sort by F1
    results_sorted = sorted(results, key=lambda x: x["f1"], reverse=True)

    print("\n===== BEST PARAM RANKING (by F1) =====")
    for rank, r in enumerate(results_sorted[:10], 1):
        print(f"{rank:02d}. F1={r['f1']:.4f}  clip={r['clip']}  alpha={r['alpha']}  [ {r['csv']} ]")
