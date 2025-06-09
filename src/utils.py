# src/utils.py

import cv2

def draw_boxes(frame, boxes, texts):
    for (box, text) in zip(boxes, texts):
        points = box.astype(int).reshape(-1, 2)
        for i in range(4):
            cv2.line(frame, tuple(points[i]), tuple(points[(i+1) % 4]), (0, 255, 0), 2)
        cv2.putText(frame, text, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame
