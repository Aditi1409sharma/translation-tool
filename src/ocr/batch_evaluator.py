import os
import re
import unicodedata
from tqdm import tqdm
from difflib import SequenceMatcher


GT_DIR = "C:/Users/aditi/GT"
CSV_DIR = "outputs_grid_search"
START_IDX = 6001
END_IDX = 7000


# --- Text normalization ---
def normalize_text(s):
    s = s.lower().strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'\s+', '', s)
    return s


# --- Load GT Japanese text for one image ---
def load_gt_japanese(gt_path):
    jp_texts = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            lang = parts[8].strip()
            text = parts[9].strip()
            if lang.lower() == "japanese":
                jp_texts.append(normalize_text(text))
    return jp_texts


# --- Fuzzy similarity ratio ---
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# --- Evaluate one CSV ---
def evaluate_csv(csv_file, similarity_threshold=0.75):
    TP, FP, FN = 0, 0, 0


    for idx in tqdm(range(START_IDX, END_IDX + 1), desc=f"Evaluating {os.path.basename(csv_file)}"):
        img_name = f"tr_img_{idx:05d}"
        gt_path = os.path.join(GT_DIR, img_name + ".txt")


        if not os.path.exists(gt_path):
            continue


        # --- Load GT texts ---
        gt_jp_texts = load_gt_japanese(gt_path)
        gt_matched = [False] * len(gt_jp_texts)


        # --- Find CSV row ---
        with open(csv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]  # skip header
            row = next((l for l in lines if l.startswith(img_name)), None)
            if row is None:
                # No OCR output for this image, all GT are false negatives
                FN += len(gt_jp_texts)
                continue


            row_parts = row.strip().split(",")
            if len(row_parts) < 4:
                FN += len(gt_jp_texts)
                continue


            jp_texts_str = ",".join(row_parts[3:]).strip().strip('"')
            if not jp_texts_str:
                FN += len(gt_jp_texts)
                continue


            # --- Parse OCR Japanese texts ---
            detected_texts = []
            for part in jp_texts_str.split("|"):
                part = part.strip()
                if not part:
                    continue
                text_only = part.split("(")[0].strip()
                if text_only:
                    detected_texts.append(normalize_text(text_only))


            # --- Match detected texts to GT with fuzzy matching ---
            for dt in detected_texts:
                match_found = False
                for i, gt in enumerate(gt_jp_texts):
                    if not gt_matched[i]:
                        sim_score = similar(dt, gt)
                        if sim_score >= similarity_threshold:
                            gt_matched[i] = True
                            TP += 1
                            match_found = True
                            break
                if not match_found:
                    FP += 1


            # --- Remaining GT unmatched count as FN ---
            FN += gt_matched.count(False)


    # --- Metrics ---
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


    print(f"\n[{os.path.basename(csv_file)}]")
    print(f"TP = {TP}, FP = {FP}, FN = {FN}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1 Score  = {f1:.4f}\n")


# --- Main: evaluate all CSVs ---
if __name__ == "__main__":
    all_csvs = [os.path.join(CSV_DIR, f) for f in os.listdir(CSV_DIR) if f.endswith(".csv")]


    print(f"Found {len(all_csvs)} CSV files to evaluate.\n")


    for csv_file in all_csvs:
        evaluate_csv(csv_file, similarity_threshold=0.75)





