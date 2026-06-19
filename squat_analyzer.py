import numpy as np

# Índices COCO (YOLOv8-pose)
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP,      R_HIP      = 11, 12
L_KNEE,     R_KNEE     = 13, 14
L_ANKLE,    R_ANKLE    = 15, 16

CONF_MIN   = 0.4
UP_ANGLE   = 160   # de pie
DOWN_ANGLE = 95    # profundidad de sentadilla


class SquatAnalyzer:
    """
    Máquina de estados: None → up → squatting → down → squatting → up → …

    "baja_mas" solo se emite cuando el usuario vuelve a "up" sin haber
    pasado por "down" (abortó la sentadilla sin llegar a profundidad).
    """

    def __init__(self):
        self.counter = 0
        self.stage   = None   # None | "up" | "squatting" | "down"
        self._reached_depth = False

    def _pt(self, lm, idx):
        if idx not in lm:
            return None
        x, y, c = lm[idx]
        return (x, y) if c >= CONF_MIN else None

    def _angle(self, a, b, c):
        a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
        ba, bc  = a - b, c - b
        cos     = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    def _knee_angle(self, lm, side):
        hip   = self._pt(lm, L_HIP   if side == 'left' else R_HIP)
        knee  = self._pt(lm, L_KNEE  if side == 'left' else R_KNEE)
        ankle = self._pt(lm, L_ANKLE if side == 'left' else R_ANKLE)
        if hip and knee and ankle:
            return self._angle(hip, knee, ankle)
        return None

    def _torso_angle(self, lm):
        """Inclinación del torso respecto a la vertical (0° = recto)."""
        ls = self._pt(lm, L_SHOULDER)
        rs = self._pt(lm, R_SHOULDER)
        lh = self._pt(lm, L_HIP)
        rh = self._pt(lm, R_HIP)
        if not all([ls, rs, lh, rh]):
            return None
        shoulder = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip      = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        dx = shoulder[0] - hip[0]
        dy = hip[1] - shoulder[1]
        return float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-8))))

    def analyze(self, landmarks):
        left  = self._knee_angle(landmarks, 'left')
        right = self._knee_angle(landmarks, 'right')

        if left and right:
            knee_angle = (left + right) / 2
        else:
            knee_angle = left or right

        torso_angle = self._torso_angle(landmarks)
        feedback    = None

        if knee_angle is not None:
            if   knee_angle > UP_ANGLE:   new_stage = "up"
            elif knee_angle < DOWN_ANGLE: new_stage = "down"
            else:                         new_stage = "squatting"

            if new_stage != self.stage:

                if new_stage == "up":
                    if self.stage == "squatting" and not self._reached_depth:
                        feedback = "baja_mas"
                    self._reached_depth = False

                elif new_stage == "down" and self.stage in ("squatting", "up", None):
                    if not self._reached_depth:
                        self._reached_depth = True
                        self.counter += 1
                        if torso_angle is not None and torso_angle > 45:
                            feedback = "espalda_recta"
                        elif self.counter <= 20:
                            feedback = f"rep_{self.counter}"
                        else:
                            feedback = "buen_rep"

                self.stage = new_stage

        return {
            "knee_angle":  knee_angle,
            "torso_angle": torso_angle,
            "stage":       self.stage,
            "counter":     self.counter,
            "feedback":    feedback,
        }
