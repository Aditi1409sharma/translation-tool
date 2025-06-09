from paddleocr import PaddleOCR, draw_ocr
import cv2

ocr = PaddleOCR(use_angle_cls=True, lang='japan', det=True, rec=True, structure=False)

ocr_result = ocr.ocr('tests/images/frame_original.png', cls=True)

print(f"OCR result: {ocr_result}")
# if ocr_result:
#     for line in ocr_result:
#         for word_info in line:
#             _, (text, confidence) = word_info
#             print(f"{text} (Confidence: {confidence:.2f})")
# else:
#     print("No text detected.")
