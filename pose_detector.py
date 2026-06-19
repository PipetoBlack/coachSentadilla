import cv2
import torch
from ultralytics import YOLO

# COCO 17-keypoint indices
KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,  'right_eye': 2,
    'left_ear': 3,  'right_ear': 4,
    'left_shoulder': 5,  'right_shoulder': 6,
    'left_elbow': 7,     'right_elbow': 8,
    'left_wrist': 9,     'right_wrist': 10,
    'left_hip': 11,      'right_hip': 12,
    'left_knee': 13,     'right_knee': 14,
    'left_ankle': 15,    'right_ankle': 16,
}

SKELETON = [
    (5, 6),  (5, 7),  (7, 9),   (6, 8),  (8, 10),   # brazos
    (5, 11), (6, 12), (11, 12),                        # torso
    (11, 13),(13, 15),(12, 14), (14, 16),              # piernas
]

class PoseDetector:
    def __init__(self, model='yolov8m-pose.pt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[PoseDetector] Usando: {self.device.upper()}")

        self.model = YOLO(model)
        self.conf_threshold = 0.4
        self._landmarks = {}

    def detect(self, frame):
        results = self.model(frame, device=self.device, verbose=False)[0]
        self._landmarks = self._extract(results)
        return self._landmarks

    def _extract(self, result):
        if result.keypoints is None or len(result.keypoints.data) == 0:
            return {}

        # Persona con mayor confianza de detección
        best = result.boxes.conf.argmax().item() if result.boxes is not None else 0
        kp = result.keypoints.data[best]  # (17, 3) -> x, y, conf

        return {
            idx: (int(kp[idx, 0].item()), int(kp[idx, 1].item()), float(kp[idx, 2].item()))
            for idx in range(len(kp))
        }

    def draw_landmarks(self, frame):
        lm = self._landmarks
        if not lm:
            return frame

        for a, b in SKELETON:
            if a in lm and b in lm:
                xa, ya, ca = lm[a]
                xb, yb, cb = lm[b]
                if ca > self.conf_threshold and cb > self.conf_threshold:
                    cv2.line(frame, (xa, ya), (xb, yb), (0, 255, 0), 2)

        for idx, (x, y, conf) in lm.items():
            if conf > self.conf_threshold:
                cv2.circle(frame, (x, y), 5, (255, 100, 0), -1)

        return frame

    def get_landmarks(self):
        return self._landmarks
