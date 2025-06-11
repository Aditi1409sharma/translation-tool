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

# --- Preprocessing Function ---
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    return denoised

# --- OCR & Save Function ---
def run_final_ocr(image_np, image_name="final_output"):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Run OCR
    result = ocr.ocr(image_np, cls=True)

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

    # Draw and save image with bounding boxes
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    image_with_boxes = draw_ocr(image_pil, boxes, txts, scores, font_path='C:\Windows\Fonts\meiryo.ttc')

    prefix = "output_steps/"

    image_with_boxes_pil = Image.fromarray(image_with_boxes)  
    image_with_boxes_pil.save(f"{prefix}image_boxed.jpg")


    # Save results to CSV
    with open(os.path.join(OUTPUT_DIR, f"{image_name}_results.csv"), "w", encoding="utf-8", newline="") as f:
        f.write("Text,Confidence\n")
        for t, s in zip(txts, scores):
            f.write(f"{t},{s:.2f}\n")

if __name__ == "__main__":
    image_path = "tests/images/frame_original1.jpg"
    original_img = cv2.imread(image_path)
    preprocessed_img = preprocess_image(original_img)
    run_final_ocr(preprocessed_img, image_name="preprocessed_final")