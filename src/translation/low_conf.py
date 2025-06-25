# import pandas as pd

# # Load the evaluation results
# df = pd.read_csv("translation_comparison_summary.csv")

# # # Load detailed comparison file if available
# # details_df = pd.read_csv("translation_comparison_detailed.csv")

# # Join on image name if needed
# merged_df = df

# # Pick a few samples with the lowest fuzzy/BLEU scores
# low_fuzzy_samples = merged_df.sort_values("Fuzzy_Avg").head(5)
# low_bleu_samples = merged_df.sort_values("BLEU_Avg").head(5)

# # Show for inspection
# print("\n--- Lowest Fuzzy Match Samples ---")
# print(low_fuzzy_samples[["Image","GT_Count","Matched_85+","Fuzzy_Avg","BLEU_Avg"]])

# print("\n--- Lowest BLEU Score Samples ---")
# print(low_bleu_samples[["Image","GT_Count","Matched_85+","Fuzzy_Avg","BLEU_Avg"]])
# lowest_fuzzy = df.nsmallest(5, "Fuzzy_Avg")
# lowest_bleu = df.nsmallest(5, "BLEU_Avg")

# # Combine and drop duplicates
# combined_low_scores = pd.concat([lowest_fuzzy, lowest_bleu]).drop_duplicates()

# # Save to CSV
# combined_low_scores.to_csv("low_score_translation_samples.csv", index=False)

# print("✅ Saved to low_score_translation_samples.csv")


import pandas as pd
import cv2
import os

# --- Paths ---
IMAGE_DIR = "C:/Users/aditi/ImagesPart2"
GT_DIR = "C:/Users/aditi/GT_Trans"  # Folder containing per-image translated GT files
OCR_FILE = "3_lower_thresh_2/translated_boxes_clip1.5_alpha0.9_scale2.csv"
BAD_IMAGE_LIST = "low_score_translation_samples.csv"
OUTPUT_DIR = "translation_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load OCR results and bad images list ---
ocr_df = pd.read_csv(OCR_FILE)
bad_df = pd.read_csv(BAD_IMAGE_LIST)

# --- Clean image IDs ---
bad_image_ids = bad_df["Image"].apply(lambda x: x.replace(".txt", "")).tolist()

# --- Function to draw bounding boxes ---
def draw_boxes(img, df, color, label_col='Translated_Text'):
    for _, row in df.iterrows():
        try:
            x_min = min(row[["X1", "X2", "X3", "X4"]])
            y_min = min(row[["Y1", "Y2", "Y3", "Y4"]])
            x_max = max(row[["X1", "X2", "X3", "X4"]])
            y_max = max(row[["Y1", "Y2", "Y3", "Y4"]])
        except:
            continue

        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        label = str(row.get(label_col, ''))
        if label and label != '###':
            cv2.putText(img, label, (int(x_min), int(y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

# --- Load GT file for an image ---
def load_gt_for_image(image_id):
    gt_path = os.path.join(GT_DIR, f"{image_id}.txt")
    if not os.path.exists(gt_path):
        print(f"[⚠️] GT file missing: {gt_path}")
        return pd.DataFrame()

    data = []
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            x1, y1, x2, y2, x3, y3, x4, y4, script, text = parts[:10]
            if script != "Japanese":
                continue
            data.append({
                "X1": float(x1), "Y1": float(y1),
                "X2": float(x2), "Y2": float(y2),
                "X3": float(x3), "Y3": float(y3),
                "X4": float(x4), "Y4": float(y4),
                "Translated_Text": text.strip()
            })
    return pd.DataFrame(data)

# --- Visualize GT vs OCR ---
def visualize(image_id):
    img_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[⚠️] Image not found: {img_path}")
        return

    gt_entries = load_gt_for_image(image_id)
    ocr_entries = ocr_df[ocr_df['Image_Name'] == image_id]

    gt_img = draw_boxes(img.copy(), gt_entries, color=(0, 255, 0))   # Green for GT
    ocr_img = draw_boxes(img.copy(), ocr_entries, color=(255, 0, 0)) # Red for OCR

    combined = cv2.hconcat([gt_img, ocr_img])
    out_path = os.path.join(OUTPUT_DIR, f"{image_id}_compare.jpg")
    cv2.imwrite(out_path, combined)
    print(f"[✅] Saved: {out_path}")

# --- Run visualization on bad images ---
print(f"📉 Visualizing {len(bad_image_ids)} poorly translated images...")
for img_id in bad_image_ids:
    visualize(img_id)
