import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GestureRecognizer:

    def __init__(self):
        self.detector = HandDetector(maxHands=2, detectionCon=0.6)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_face(self, frame) -> Optional[tuple]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        return tuple(faces[0])

    def is_hand_near_face(self, hand, face_bbox, x_factor: float = 1.8, y_factor: float = 1.8) -> bool:

        if face_bbox is None:
            return False

        fx, fy, fw, fh = face_bbox
        face_cx = fx + fw // 2
        face_cy = fy + fh // 2

        bbox = hand.get("bbox", None)
        lmList = hand.get("lmList", [])

        if bbox:
            x, y, w, h = bbox
            hand_cx = x + w // 2
            hand_cy = y + h // 2
        elif lmList:
            hand_cx = (lmList[0][0] + lmList[12][0]) // 2
            hand_cy = (lmList[0][1] + lmList[12][1]) // 2
        else:
            return False

        return (
            abs(hand_cx - face_cx) <= fw * x_factor
            and abs(hand_cy - face_cy) <= fh * y_factor
        )

    def detect_thinking_gesture(self, hand, face_bbox=None) -> bool:
        lmList = hand.get("lmList", [])
        if not lmList:
            return False

        if face_bbox is None or not self.is_hand_near_face(hand, face_bbox, x_factor=2.0, y_factor=2.0):
            return False

        wrist_y = lmList[0][1]
        middle_finger_y = lmList[12][1]
        base_condition = middle_finger_y < wrist_y - 70

        return base_condition

    def detect_pointing_up(self, hand, face_bbox=None) -> bool:
        fingers = self.detector.fingersUp(hand)
        is_pointing = fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0
        if not is_pointing:
            return False

        if face_bbox is None or not self.is_hand_near_face(hand, face_bbox, x_factor=2.5, y_factor=2.5):
            return False

        return True

    def detect_surprised_gesture(self, hand, face_bbox=None) -> bool:
        fingers = self.detector.fingersUp(hand)
        is_surprised = sum(fingers) >= 4
        if not is_surprised:
            return False

        if face_bbox is None or not self.is_hand_near_face(hand, face_bbox, x_factor=2.2, y_factor=2.2):
            return False

        return True

    def recognize_gesture(self, hands, frame, face_bbox: Optional[tuple] = None) -> Optional[str]:
        if not hands:
            return None

        if face_bbox is not None and len(hands) > 1:
            fx, fy, fw, fh = face_bbox
            face_cx = fx + fw // 2
            face_cy = fy + fh // 2

            def hand_center(h):
                bbox = h.get("bbox", None)
                lmList = h.get("lmList", [])
                if bbox:
                    x, y, w, h_w = bbox
                    return x + w // 2, y + h_w // 2
                if lmList:
                    return (
                        (lmList[0][0] + lmList[12][0]) // 2,
                        (lmList[0][1] + lmList[12][1]) // 2,
                    )
                return None

            min_dist = None
            chosen_hand = hands[0]
            for h in hands:
                center = hand_center(h)
                if center is None:
                    continue
                cx, cy = center
                dist_sq = (cx - face_cx) ** 2 + (cy - face_cy) ** 2
                if min_dist is None or dist_sq < min_dist:
                    min_dist = dist_sq
                    chosen_hand = h
            hand = chosen_hand
        else:
            hand = hands[1] if len(hands) > 1 else hands[0]

        if self.detect_thinking_gesture(hand, face_bbox=face_bbox):
            return "thinking"
        if self.detect_pointing_up(hand, face_bbox=face_bbox):
            return "pointing_up"
        if self.detect_surprised_gesture(hand, face_bbox=face_bbox):
            return "surprised"

        return "neutral"


class MemeOverlay:

    def __init__(self, meme_dir: Path = Path("memes")):
        self.meme_dir = meme_dir
        self.memes: Dict[str, np.ndarray] = {}
        self.overlay_size = (200, 200)
        self.overlay_position = (20, 20)
        self.alpha = 0.0
        self.target_alpha = 0.0
        self.fade_speed = 0.1
        self.current_meme: Optional[str] = None

    def load_memes(self):
        meme_files = {
            "thinking": "monkey_pointing.jpg",
            "pointing_up": "monkey_thinking.jpg",
            "neutral": "monkey_neutral.jpg",
            "surprised": "monkey_surprised.jpg",
        }

        for gesture, filename in meme_files.items():
            filepath = self.meme_dir / filename
            if filepath.exists():
                img = cv2.imread(str(filepath))
                if img is not None:
                    img = cv2.resize(img, self.overlay_size)
                    self.memes[gesture] = img
                    logger.info(f"Loaded meme: {gesture}")
            else:
                self.memes[gesture] = self.create_placeholder(gesture)

    def create_placeholder(self, gesture: str) -> np.ndarray:
        img = np.ones((self.overlay_size[1], self.overlay_size[0], 3), dtype=np.uint8) * 200
        text = gesture.replace("_", " ").title()
        cv2.putText(img, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)
        return img

    def update_gesture(self, gesture: Optional[str]):
        if gesture != self.current_meme:
            self.current_meme = gesture
            self.target_alpha = 1.0 if gesture else 0.0

    def apply_overlay(self, frame: np.ndarray) -> np.ndarray:
        if abs(self.alpha - self.target_alpha) > 0.01:
            self.alpha += (self.target_alpha - self.alpha) * self.fade_speed

        if self.alpha < 0.05 or not self.current_meme:
            return frame

        meme = self.memes.get(self.current_meme)
        if meme is None:
            return frame

        x, y = self.overlay_position
        h, w = meme.shape[:2]

        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            return frame

        roi = frame[y : y + h, x : x + w]
        blended = cv2.addWeighted(roi, 1 - self.alpha, meme, self.alpha, 0)
        frame[y : y + h, x : x + w] = blended

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        return frame


class GestureMemeApp:

    def __init__(self):
        self.recognizer = GestureRecognizer()
        self.overlay = MemeOverlay()
        self.cap = None

    def initialize(self) -> bool:
        logger.info("Initializing application...")

        self.overlay.meme_dir.mkdir(exist_ok=True)
        self.overlay.load_memes()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Failed to open camera")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        logger.info("Initialization complete")
        return True

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, 1)

        face_bbox = self.recognizer.detect_face(frame)

        hands, frame = self.recognizer.detector.findHands(frame, draw=False)

        gesture = self.recognizer.recognize_gesture(hands, frame, face_bbox=face_bbox)

        if gesture is None:
            gesture = "neutral"

        self.overlay.update_gesture(gesture)
        frame = self.overlay.apply_overlay(frame)

        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

        self.draw_ui(frame, gesture)

        return frame

    def draw_ui(self, frame: np.ndarray, gesture: Optional[str]):
        cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        if not self.initialize():
            return

        logger.info("Starting main loop. Press 'q' to quit.")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = self.process_frame(frame)
                cv2.imshow("Gesture Meme Overlay", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    app = GestureMemeApp()
    app.run()


if __name__ == "__main__":
    main()