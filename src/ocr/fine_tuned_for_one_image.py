import cv2
import os
import csv
from ocr_infer import preprocess_image, ocr
import numpy as np

CONFIDENCE_THRESHOLD = 0.7
IMAGE_NAME = "frame_original2.jpg"
IMAGE_PATH = os.path.abspath(os.path.join("tests/images", IMAGE_NAME))
OUTPUT_CSV = os.path.join("outputs_steps", "single_image_rotations.csv")
VIS_DIR = os.path.join("outputs_steps", "visualizations")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

def rotate_image(img, angle):
    if angle == 0:
        return img
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

def draw_boxes(image, ocr_result, angle):
    for line in ocr_result:
        for box, (text, score) in line:
            box = [[int(p[0]), int(p[1])] for p in box]
            cv2.polylines(image, [np.array(box)], isClosed=True, color=(0, 0, 255), thickness=10)  # Red box
            label = f"{text} ({score:.2f})"
            x, y = box[0]
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return image


def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Image not found: {IMAGE_PATH}")
        return

    color_img = cv2.imread(IMAGE_PATH)
    if color_img is None:
        print(f"[ERROR] Could not read image.")
        return

    all_results = []

    for angle in [0, 90, 180, 270]:
        # Rotate color image first
        rotated_color = rotate_image(color_img, angle)

        # Preprocess the rotated image (e.g., grayscale, normalization, etc.)
        preprocessed = preprocess_image(rotated_color)

        # Perform OCR on the preprocessed image
        result = ocr.ocr(preprocessed, cls=True)

        # Draw boxes on the preprocessed image (converted to BGR for color drawing)
        vis_img = cv2.cvtColor(preprocessed.copy(), cv2.COLOR_GRAY2BGR)
        vis_img = draw_boxes(vis_img, result) 

        # Save visualized image
        vis_path = os.path.join(VIS_DIR, f"{os.path.splitext(IMAGE_NAME)[0]}_rot{angle}.jpg")
        cv2.imwrite(vis_path, vis_img)

        # Collect OCR results
        for line in result:
            for box, (text, score) in line:
                all_results.append({
                    "rotation": angle,
                    "text": text,
                    "confidence": round(score, 4),
                    "bbox": box
                })

    # Write results to CSV
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rotation", "text", "confidence", "bbox"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"✅ OCR results written to {OUTPUT_CSV}")
    print(f"📸 Visualizations saved in: {VIS_DIR}")
