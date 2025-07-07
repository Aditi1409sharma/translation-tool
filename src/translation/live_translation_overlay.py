# # live_translation_overlay.py
# import cv2
# import time
# from ocr_utils import perform_ocr
# from translate_utils import translate_text

# # --- Overlay Helper ---
# def overlay_text(img, box, jp_text, en_text):
#     (x1, y1), (x2, y2) = box
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     text = f"{jp_text} → {en_text}"
#     cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# # --- Main Loop ---
# def main():
#     cap = cv2.VideoCapture(0)  # Use default webcam
#     if not cap.isOpened():
#         print("❌ Could not access camera.")
#         return

#     print("📷 Starting live OCR + translation... Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         start_time = time.time()

#         # Run OCR
#         boxes, jp_texts = perform_ocr(frame)

#         # Translate detected texts
#         en_texts = [translate_text(txt) for txt in jp_texts]

#         # Draw on frame
#         for box, jp, en in zip(boxes, jp_texts, en_texts):
#             overlay_text(frame, box, jp, en)

#         # Show FPS
#         fps = 1.0 / (time.time() - start_time)
#         cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

#         # Display result
#         cv2.imshow("Live Translation Overlay", frame)

#         # Exit on 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
from paddleocr import PaddleOCR
from translate_utils import translate_text
import threading
import time

ocr = PaddleOCR(use_angle_cls=True, lang='japan')
scale = 2

cap = cv2.VideoCapture(0)

# Cache translations to speed up repeated translations
translation_cache = {}

# Shared variables for threading
latest_frame = None
processed_frame = None
lock = threading.Lock()
frame_idx = 0

def process_frame_loop():
    global processed_frame, latest_frame, translation_cache

    while True:
        if latest_frame is None:
            time.sleep(0.02)
            continue

        frame_copy = None
        with lock:
            frame_copy = latest_frame.copy()

        # Preprocess like before
        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(gray_clahe, h=10)
        kernel = np.array([[0, -1, 0], [-1, 4.5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        dilated = cv2.dilate(sharpened, np.ones((1, 1), np.uint8), iterations=1)
        upscaled = cv2.resize(dilated, (frame_copy.shape[1] * scale, frame_copy.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        preprocessed = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)

        result = ocr.ocr(preprocessed, cls=True)

        overlay = frame_copy.copy()

        for line in result:
            for box_info in line:
                box, (text, score) = box_info
                if score < 0.3:
                    continue

                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                x_min = int(min(xs) / scale)
                y_min = int(min(ys) / scale)
                x_max = int(max(xs) / scale)
                y_max = int(max(ys) / scale)

                # Blur ROI
                roi = overlay[y_min:y_max, x_min:x_max]
                blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
                overlay[y_min:y_max, x_min:x_max] = blurred_roi

                # Translate with caching
                if text not in translation_cache:
                    translation_cache[text] = translate_text(text)
                translated_text = translation_cache[text]

                # Draw translated text
                font_scale = 0.6  # Smaller text
                font_color = (0, 0, 255)  # Red in BGR
                thickness = 1

                cv2.putText(overlay, translated_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness, cv2.LINE_AA)


                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        with lock:
            processed_frame = overlay

        time.sleep(0.2)  # Sleep to control processing rate (adjust for performance)

# Start processing thread
threading.Thread(target=process_frame_loop, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with lock:
        latest_frame = frame.copy()
        display_frame = processed_frame if processed_frame is not None else frame

    # cv2.imshow("Live Translation Overlay", display_frame)
    resized_overlay = cv2.resize(display_frame, None, fx=1.5, fy=1.5)  # Scale 1.5x bigger
    cv2.imshow("Live Translation", resized_overlay)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
