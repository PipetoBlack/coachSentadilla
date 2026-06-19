import cv2
from pose_detector import PoseDetector
from squat_analyzer import SquatAnalyzer
from voice_coach import VoiceCoach

CAMERA_INDEX = 1   # cambia a 0 si usas la cámara integrada


def draw_ui(frame, result):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (310, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"Reps: {result['counter']}",
                (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)

    STAGE_ES = {"up": "arriba", "squatting": "bajando", "down": "abajo"}
    stage_text = STAGE_ES.get(result['stage'], "---")
    cv2.putText(frame, f"Estado: {stage_text}",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if result['knee_angle'] is not None:
        angle = int(result['knee_angle'])
        color = (0, 255, 0) if angle < 95 else (0, 200, 255)
        cv2.putText(frame, f"Rodilla: {angle} grad",
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    if result['torso_angle'] is not None:
        tangle = int(result['torso_angle'])
        color  = (0, 255, 0) if tangle < 35 else (0, 80, 255)
        cv2.putText(frame, f"Torso:   {tangle} grad",
                    (20, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return frame


def main():
    cap      = cv2.VideoCapture(CAMERA_INDEX)
    detector = PoseDetector(model='yolov8m-pose.pt')
    analyzer = SquatAnalyzer()
    coach    = VoiceCoach()

    coach.say("inicio")
    last_feedback = None

    cv2.namedWindow("Squat Coach", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        landmarks = detector.detect(frame)
        detector.draw_landmarks(frame)

        result = analyzer.analyze(landmarks)

        if result["feedback"] and result["feedback"] != last_feedback:
            coach.say(result["feedback"])
            last_feedback = result["feedback"]
        if result["feedback"] is None:
            last_feedback = None

        draw_ui(frame, result)
        cv2.imshow("Squat Coach", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
