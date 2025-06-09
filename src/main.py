# src/main.py

import cv2
from ocr_engine import extract_text
from translator import translate_text
from utils import draw_boxes
from config import FRAME_WIDTH, FRAME_HEIGHT

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, jpn_texts = extract_text(frame)
        eng_texts = [translate_text(t) for t in jpn_texts]
        annotated_frame = draw_boxes(frame, boxes, eng_texts)

        cv2.imshow("Japanese to English OCR", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
