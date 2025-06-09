# src/ocr_engine.py

from paddleocr import PaddleOCR
from config import OCR_LANG, OCR_USE_GPU

ocr_model = PaddleOCR(use_angle_cls=True, lang=OCR_LANG, use_gpu=OCR_USE_GPU)

def extract_text(frame):
    result = ocr_model.ocr(frame, cls=True)
    boxes = []
    texts = []
    for line in result[0]:
        box, (text, _) = line
        boxes.append(box)
        texts.append(text)
    return boxes, texts

