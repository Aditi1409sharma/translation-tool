import os
from tqdm import tqdm

GT_DIR = "C:/Users/aditi/GT"
OCR_LOG = "outputs_steps/batch_results_filtered.csv"
START_IDX = 6001
END_IDX = 7000

# --- Load OCR Results ---
ocr_dict = {}
with open(OCR_LOG, "r", encoding="utf-8") as f:
    lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(",")
        img_name = parts[0]
        jp_texts = parts[-1].strip('"').split(" | ") if len(parts) > 4 else []
        ocr_dict[img_name] = jp_texts

# --- Evaluation ---
tp, fp, fn = 0, 0, 0
for idx in tqdm(range(START_IDX, END_IDX + 1), desc="Evaluating vs GT"):
    img_name = f"tr_img_{idx:05d}"
    gt_path = os.path.join(GT_DIR, img_name + ".txt")

    if not os.path.exists(gt_path):
        print(f"[WARNING] Missing GT: {gt_path}")
        continue

    gt_texts = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            lang = parts[-2]
            text = parts[-1]
            if lang == "Japanese":
                gt_texts.append(text)

    gt_set = set(gt_texts)
    ocr_set = set(ocr_dict.get(img_name, []))

    tp += len(gt_set & ocr_set)
    fp += len(ocr_set - gt_set)
    fn += len(gt_set - ocr_set)

# --- Metrics ---
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

print(f"\n--- Evaluation ---")
print(f"TP = {tp}, FP = {fp}, FN = {fn}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1 Score  = {f1:.4f}")
