import os
import csv
from tqdm import tqdm
from transformers import pipeline
from difflib import SequenceMatcher
from collections import Counter

# --- Paths ---
RESULTS_CSV = "outputs_grid_search/results_clip1.8_alpha0.9.csv"  # change as needed
GT_TRANSLATED_DIR = "C:/Users/aditi/GT_translated"
OUTPUT_CSV = RESULTS_CSV.replace(".csv", "_comparison.csv")

# --- Load local model ---
MODEL_PATH = "C:/Users/aditi/translation_models/opus-mt-ja-en"
translator = pipeline("translation", model=MODEL_PATH)

# --- Fuzzy similarity ---
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# --- Counters for metrics ---
TP = 0
FP = 0
FN = 0

# --- Process results ---
with open(RESULTS_CSV, "r", encoding="utf-8") as fin, \
     open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fout:

    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(["Image", "Total_Boxes", "OCR_Japanese", "OCR_Translated", "GT_Translated", "Similarity", "TP", "FP", "FN"])

    for row in tqdm(reader, desc="Comparing translations"):
        img_name = row["Image"]

        # --- Load GT translated text ---
        gt_path = os.path.join(GT_TRANSLATED_DIR, img_name + ".txt")
        if not os.path.exists(gt_path):
            print(f"[WARNING] Missing GT: {gt_path}")
            continue

        gt_translated_words = []
        with open(gt_path, "r", encoding="utf-8") as gt_file:
            for line in gt_file:
                parts = line.strip().split(",")
                if len(parts) < 10:
                    continue
                lang = parts[8]
                text = parts[9]
                if lang.lower() == "japanese" and text.strip():
                    # Split GT text into words
                    gt_translated_words += text.lower().split()

        gt_counter = Counter(gt_translated_words)
        gt_text_str = " | ".join(gt_translated_words)

        # --- OCR texts ---
        ocr_jp_texts = []
        ocr_translated_words = []

        jp_field = row["Japanese_Texts"]
        if jp_field.strip():
            jp_items = jp_field.split("|")
            for item in jp_items:
                text_part = item.strip().split(" (")[0]  # remove confidence
                ocr_jp_texts.append(text_part)

                # Translate via model
                trans_out = translator(text_part)
                trans_text = trans_out[0]["translation_text"].lower()
                # Split translated text into words
                ocr_translated_words += trans_text.split()

        ocr_jp_str = " | ".join(ocr_jp_texts)
        ocr_trans_str = " | ".join(ocr_translated_words)

        # --- Compute per-image TP/FP/FN ---
        tp = 0
        fp = 0
        fn = 0

        gt_tmp_counter = gt_counter.copy()

        for word in ocr_translated_words:
            if gt_tmp_counter[word] > 0:
                tp += 1
                gt_tmp_counter[word] -= 1
            else:
                fp += 1

        fn = sum(gt_tmp_counter.values())

        # Update global counts
        TP += tp
        FP += fp
        FN += fn

        sim = similar(ocr_trans_str, gt_text_str)

        # --- Write row ---
        writer.writerow([
            img_name,
            row["Total_Boxes"],
            ocr_jp_str,
            ocr_trans_str,
            gt_text_str,
            f"{sim:.4f}",
            tp,
            fp,
            fn
        ])

# --- Final Metrics ---
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

print("\n=== FINAL METRICS ===")
print(f"TP: {TP}, FP: {FP}, FN: {FN}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"\n✅ Comparison CSV saved: {OUTPUT_CSV}")
