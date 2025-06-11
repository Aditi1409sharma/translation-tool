import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os

# --- Constants ---
CONFIDENCE_THRESHOLD = 0.5
OUTPUT_DIR = "outputs_steps"
FONT_PATH = "C:\Windows\Fonts\meiryo.ttc"  # Update to valid .ttf font path

# --- Initialize OCR ---
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# --- Image Rotation Helper ---
def rotate_image(img, angle):
    if angle == 0:
        return img
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

# --- Auto-Rotate to Maximize OCR Confidence ---
def get_best_rotation(img):
    best_score, best_angle, best_result = -1, 0, None
    best_rotated = img

    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(img, angle)
        result = ocr.ocr(rotated, cls=True)

        score_sum, count = 0, 0
        for line in result:
            for _, (_, score) in line:
                score_sum += score
                count += 1

        avg_conf = score_sum / count if count else 0
        if avg_conf > best_score:
            best_score = avg_conf
            best_angle = angle
            best_result = result
            best_rotated = rotated

    return best_rotated, best_result, best_angle, best_score

# --- Preprocessing Function ---
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    return denoised

# --- OCR & Save Function ---
def run_final_ocr(image_np, image_name="final_output"):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Try all rotations and pick best one
    best_img, result, used_angle, avg_conf = get_best_rotation(image_np)
    print(f"[INFO] Best Rotation: {used_angle}°, Avg Conf: {avg_conf:.4f}")

    boxes, txts, scores = [], [], []
    total_boxes, retained_boxes, score_sum = 0, 0, 0

    for line in result:
        for word_info in line:
            total_boxes += 1
            box, (text, score) = word_info
            if score >= CONFIDENCE_THRESHOLD:
                boxes.append(box)
                txts.append(text)
                scores.append(score)
                score_sum += score
                retained_boxes += 1

    avg_conf = score_sum / retained_boxes if retained_boxes else 0.0
    print(f"[FINAL] Total Boxes: {total_boxes}, Retained: {retained_boxes}, Avg Conf: {avg_conf:.2f}")

    # Draw boxes and save
    if len(best_img.shape) == 2:
        best_img = cv2.cvtColor(best_img, cv2.COLOR_GRAY2BGR)
    image_pil = Image.fromarray(cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB))
    image_with_boxes = draw_ocr(image_pil, boxes, txts, scores, font_path=FONT_PATH)
    output_path = os.path.join(OUTPUT_DIR, f"{image_name}_boxed.jpg")
    Image.fromarray(image_with_boxes).save(output_path)

    # Save CSV
    with open(os.path.join(OUTPUT_DIR, f"{image_name}_results.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("Text,Confidence\n")
        for t, s in zip(txts, scores):
            f.write(f"{t},{s:.2f}\n")

if __name__ == "__main__":
    image_path = "tests/images/frame_original1.jpg"
    original_img = cv2.imread(image_path)
    preprocessed_img = preprocess_image(original_img)
    run_final_ocr(preprocessed_img, image_name="preprocessed_final")