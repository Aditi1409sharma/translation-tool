# # import os
# # import pandas as pd
# # from tqdm import tqdm
# # from difflib import SequenceMatcher
# # from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# # import nltk
# # nltk.download('punkt')


# # # --- Paths ---
# # GT_TRANSLATED_FOLDER = "C:/Users/aditi/GT_trans"
# # OCR_CSV = "3_lower_thresh_2/translated_boxes_clip1.5_alpha0.9_scale2.csv"

# # # --- Load OCR Translations ---
# # ocr_df = pd.read_csv(OCR_CSV)
# # ocr_df["Image_ID"] = ocr_df["Image_Name"].str.extract(r"(\d+)$")

# # # --- Similarity Functions ---
# # def fuzzy_match(a, b):
# #     return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

# # def get_bleu(ref, hyp):
# #     ref_tokens = ref.strip().split()
# #     hyp_tokens = hyp.strip().split()
# #     if len(hyp_tokens) == 0:
# #         return 0
# #     return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1)

# # # --- Evaluate ---
# # results = []

# # for fname in tqdm(sorted(os.listdir(GT_TRANSLATED_FOLDER))[:100]):
# #     if not fname.endswith(".txt"):
# #         continue

# #     image_id = fname.split("_")[-1].split(".")[0]
# #     ocr_rows = ocr_df[ocr_df["Image_ID"] == image_id]
# #     if ocr_rows.empty:
# #         print(f"[SKIP] No OCR rows found for image ID: {image_id}")
# #         continue
# #     print(ocr_df["Image_Name"].head())
# #     print(ocr_df["Image_ID"].unique()[:5])


# #     # Load translated GT labels
# #     with open(os.path.join(GT_TRANSLATED_FOLDER, fname), encoding="utf-8") as f:
# #         gt_lines = [line.strip().split(",")[-1] for line in f.readlines()]
# #         gt_lines = [line for line in gt_lines if line not in ["###", "", "None"]]  # Clean up noise

# #     # Load translated OCR labels
# #     ocr_lines = list(ocr_rows["Translated_Text"].dropna())
# #     ocr_lines = [line.strip() for line in ocr_lines if line not in ["###", "", "None"]]

# #     matched = 0
# #     fuzzy_scores = []
# #     bleu_scores = []

# #     for gt_text in gt_lines:
# #         scores = [fuzzy_match(gt_text, ocr_text) for ocr_text in ocr_lines]
# #         bleus = [get_bleu(gt_text, ocr_text) for ocr_text in ocr_lines]
# #         if scores:
# #             best_score = max(scores)
# #             fuzzy_scores.append(best_score)
# #             bleu_scores.append(max(bleus))
# #             if best_score > 0.85:
# #                 matched += 1

# #     total = len(gt_lines)
# #     result = {
# #         "Image": fname,
# #         "GT_Count": total,
# #         "Matched_85+": matched,
# #         "Fuzzy_Avg": sum(fuzzy_scores)/len(fuzzy_scores) if fuzzy_scores else 0,
# #         "BLEU_Avg": sum(bleu_scores)/len(bleu_scores) if bleu_scores else 0
# #     }
# #     results.append(result)

# # # --- Final Output ---
# # results_df = pd.DataFrame(results)
# # results_df.to_csv("translation_comparison_summary.csv", index=False)
# # print("✅ Evaluation complete. Results saved to translation_comparison_summary.csv")
# # print(results_df.describe())





import os
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
from evaluate import load
import nltk
nltk.download('punkt')

# --- Load Evaluators ---
chrf = load("chrf")
ter = load("ter")
# comet = load("comet", module_type="metric")  # Optional: very heavy (~1GB model)

# --- Paths ---
GT_TRANSLATED_FOLDER = "C:/Users/aditi/GT_trans"
OCR_CSV = "best_version/translated_boxes_clip1.5_alpha0.9_scale2.csv"

ocr_df = pd.read_csv(OCR_CSV)
ocr_df["Image_ID"] = ocr_df["Image_Name"].str.extract(r"(\d+)$")

def fuzzy_match(a, b):
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

def load_text_lines(gt_path):
    with open(gt_path, encoding="utf-8") as f:
        lines = [line.strip().split(",")[-1] for line in f.readlines()]
        return [line for line in lines if line not in ["###", "", "None"]]

results = []

for fname in tqdm(sorted(os.listdir(GT_TRANSLATED_FOLDER))[:500]):
    if not fname.endswith(".txt"):
        continue

    image_id = fname.split("_")[-1].split(".")[0]
    gt_path = os.path.join(GT_TRANSLATED_FOLDER, fname)
    gt_lines = load_text_lines(gt_path)

    # ⚠️ Skip if GT has no valid lines
    if not gt_lines:
        continue

    ocr_rows = ocr_df[ocr_df["Image_ID"] == image_id]
    ocr_lines = list(ocr_rows["Translated_Text"].dropna())
    ocr_lines = [line.strip() for line in ocr_lines if line not in ["###", "", "None"]]

    # Aggregate scores
    fuzzy_scores, chrf_scores, ter_scores, comet_scores = [], [], [], []

    for gt_text in gt_lines:
        if not ocr_lines:
            continue

        best_fuzzy, best_chrf, best_ter, best_comet = 0, 0, 0, 0
        for ocr_text in ocr_lines:
            fuzzy = fuzzy_match(gt_text, ocr_text)
            chrf_score = chrf.compute(predictions=[ocr_text], references=[gt_text])['score']
            ter_score = ter.compute(predictions=[ocr_text], references=[gt_text])['score']
            # comet_score = comet.compute(predictions=[ocr_text], references=[gt_text], sources=[gt_text])['score']

            if fuzzy > best_fuzzy:
                best_fuzzy = fuzzy
                best_chrf = chrf_score
                best_ter = ter_score
                # best_comet = comet_score

        fuzzy_scores.append(best_fuzzy)
        chrf_scores.append(best_chrf)
        ter_scores.append(best_ter)
        comet_scores.append(best_comet)

    result = {
        "Image": fname,
        "GT_Count": len(gt_lines),
        "Fuzzy_Avg": sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0,
        "ChrF++_Avg": sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0,
        "TER_Avg": sum(ter_scores) / len(ter_scores) if ter_scores else 0,
        # "COMET_Avg": sum(comet_scores) / len(comet_scores) if comet_scores else 0,
    }
    results.append(result)

# --- Save Results ---
results_df = pd.DataFrame(results)
results_df.to_csv("translation_eval_advanced.csv", index=False)
print("✅ Evaluation complete. Results saved to translation_eval_advanced.csv")
print(results_df.describe())






















# import os
# import pandas as pd
# from tqdm import tqdm
# from difflib import SequenceMatcher
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# import nltk
# nltk.download('punkt')


# # --- Paths ---
# GT_TRANSLATED_FOLDER = "C:/Users/aditi/GT_trans"
# OCR_CSV = "3_lower_thresh_2/translated_boxes_clip1.5_alpha0.9_scale2.csv"

# # --- Load OCR Translations ---
# ocr_df = pd.read_csv(OCR_CSV)
# ocr_df["Image_ID"] = ocr_df["Image_Name"].str.extract(r"(\d+)$")

# # --- Similarity Functions ---
# def fuzzy_match(a, b):
#     return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

# def get_bleu(ref, hyp):
#     ref_tokens = ref.strip().split()
#     hyp_tokens = hyp.strip().split()
#     if len(hyp_tokens) == 0:
#         return 0
#     return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1)

# # --- Thresholds to evaluate (lowest ones) ---
# MATCH_THRESHOLDS = [0.65, 0.7]

# # --- Evaluate ---
# results = []

# for fname in tqdm(sorted(os.listdir(GT_TRANSLATED_FOLDER))[:100]):
#     if not fname.endswith(".txt"):
#         continue

#     image_id = fname.split("_")[-1].split(".")[0]
#     ocr_rows = ocr_df[ocr_df["Image_ID"] == image_id]
#     if ocr_rows.empty:
#         print(f"[SKIP] No OCR rows found for image ID: {image_id}")
#         continue

#     # Load translated GT labels
#     with open(os.path.join(GT_TRANSLATED_FOLDER, fname), encoding="utf-8") as f:
#         gt_lines = [line.strip().split(",")[-1] for line in f.readlines()]
#         gt_lines = [line for line in gt_lines if line not in ["###", "", "None"]]  # Clean up noise

#     # Load translated OCR labels
#     ocr_lines = list(ocr_rows["Translated_Text"].dropna())
#     ocr_lines = [line.strip() for line in ocr_lines if line not in ["###", "", "None"]]

#     fuzzy_scores = []
#     bleu_scores = []

#     # Counters for each threshold
#     matched_counts = {thresh: 0 for thresh in MATCH_THRESHOLDS}

#     for gt_text in gt_lines:
#         scores = [fuzzy_match(gt_text, ocr_text) for ocr_text in ocr_lines]
#         bleus = [get_bleu(gt_text, ocr_text) for ocr_text in ocr_lines]

#         if scores:
#             best_score = max(scores)
#             best_bleu = max(bleus)
#             fuzzy_scores.append(best_score)
#             bleu_scores.append(best_bleu)

#             for thresh in MATCH_THRESHOLDS:
#                 if best_score > thresh:
#                     matched_counts[thresh] += 1

#     total = len(gt_lines)
#     result = {
#         "Image": fname,
#         "GT_Count": total,
#         "Fuzzy_Avg": sum(fuzzy_scores)/len(fuzzy_scores) if fuzzy_scores else 0,
#         "BLEU_Avg": sum(bleu_scores)/len(bleu_scores) if bleu_scores else 0,
#     }

#     # Add matched counts for each threshold
#     for thresh in MATCH_THRESHOLDS:
#         result[f"Matched_>{thresh}"] = matched_counts[thresh]

#     results.append(result)

# # --- Final Output ---
# results_df = pd.DataFrame(results)
# results_df.to_csv("translation_comparison_summary_low_thresholds.csv", index=False)
# print("✅ Evaluation complete. Results saved to translation_comparison_summary_low_thresholds.csv")
# print(results_df.describe())

