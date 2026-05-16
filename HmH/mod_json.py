import cv2
import mediapipe as mp
import time
import json

# Landmark names based on the user's provided screenshot
LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.3, track_con=0.3):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.results = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            model_complexity=1,  # Confirmed stable setting
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_all_positions(self, img):
        all_hand_info = []

        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                hand_data = []

                hand_label = handedness.classification[0].label
                hand_score = handedness.classification[0].score

                h, w, c = img.shape

                # Dictionary to hold the 21 points reported by MediaPipe (only successful ones)
                landmarks_map = {}

                # Fill the map with data for points that MediaPipe reports
                for index, lm in enumerate(hand_landmarks.landmark):
                    # Convert to pixel coordinates (integer)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks_map[index] = (cx, cy, lm.z)

                # Iterate through the full 21-point range to save data consistently
                for index in range(21):
                    # Get reported coordinates, or default to (0, 0, 0) if the index was not reported
                    cx, cy, z = landmarks_map.get(index, (0, 0, 0))

                    # Store data for all 21 points
                    hand_data.append({
                        "index": index,
                        "x": cx,
                        "y": cy,
                        "z": z
                    })

                all_hand_info.append({
                    "label": hand_label,
                    "data": hand_data,
                    "score": hand_score
                })

        return all_hand_info

    def landmarks_to_json(self, all_hand_info):
        frame_keypoints = {"keypoints": []}

        if not all_hand_info:
            return frame_keypoints

        for hand_info in all_hand_info:
            hand_label = hand_info["label"]
            hand_data = hand_info["data"]
            hand_score = hand_info["score"]

            for point in hand_data:
                index = point["index"]
                x = point["x"]
                y = point["y"]
                z = point["z"]

                joint_name = LANDMARK_NAMES[index]

                keypoint = {
                    "joint": f"{hand_label.upper()}_{joint_name}",
                    "x": x,
                    "y": y,
                    "z": z,
                    "confidence": hand_score
                }
                frame_keypoints["keypoints"].append(keypoint)

        return frame_keypoints