import os
import pandas as pd
from difflib import SequenceMatcher

GT_TRANSLATED_FOLDER = "C:/Users/aditi/GT_trans"
OCR_CSV = "3_lower_thresh_2_500/translated_boxes_clip1.5_alpha0.9_scale2.csv"

ocr_df = pd.read_csv(OCR_CSV)
ocr_df["Image_ID"] = ocr_df["Image_Name"].str.extract(r"(\d+)$")

def fuzzy_match(a, b):
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

def load_text_lines(gt_path):
    with open(gt_path, encoding="utf-8") as f:
        lines = [line.strip().split(",")[-1] for line in f.readlines()]
        return [line for line in lines if line not in ["###", "", "None"]]

# Load your evaluation results from CSV
results_df = pd.read_csv("translation_eval_advanced.csv")

# Pick top 3 highest and lowest fuzzy avg images
top_n = 10
top_images = results_df.sort_values("Fuzzy_Avg", ascending=False).head(top_n)
bottom_images = results_df.sort_values("Fuzzy_Avg").head(top_n)

def print_comparison(image_list, label):
    print(f"\n--- {label} ---\n")
    for _, row in image_list.iterrows():
        fname = row['Image']
        image_id = fname.split("_")[-1].split(".")[0]
        gt_path = os.path.join(GT_TRANSLATED_FOLDER, fname)
        
        gt_lines = load_text_lines(gt_path)
        ocr_rows = ocr_df[ocr_df["Image_ID"] == image_id]
        ocr_lines = list(ocr_rows["Translated_Text"].dropna())
        ocr_lines = [line.strip() for line in ocr_lines if line not in ["###", "", "None"]]

        print(f"Image: {fname}")
        print(f"Fuzzy_Avg: {row['Fuzzy_Avg']:.3f}")
        print(f"GT Text Lines ({len(gt_lines)}):")
        for line in gt_lines:
            print(f"  - {line}")
        print(f"OCR Text Lines ({len(ocr_lines)}):")
        for line in ocr_lines:
            print(f"  - {line}")
        print("\n" + "-"*50 + "\n")

print_comparison(top_images, "High Fuzzy Score Images")
print_comparison(bottom_images, "Low Fuzzy Score Images")
