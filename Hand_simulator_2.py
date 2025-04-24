import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Finger connections
finger_indices = [list(range(0, 5)), list(range(5, 9)), list(range(9, 13)), list(range(13, 17)), list(range(17, 21))]
palm_connections = [(0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]

# Buffer to store wrist positions for movement detection
right_wrist_history = deque(maxlen=10)

def draw_hand_skeleton(landmarks, white_img, center_point):
    h, w = white_img.shape[:2]
    pts = []

    for lm in landmarks.landmark:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        pts.append((cx, cy))

    mean_x = int(np.mean([p[0] for p in pts]))
    mean_y = int(np.mean([p[1] for p in pts]))

    shift_x = center_point[0] - mean_x
    shift_y = center_point[1] - mean_y
    shifted_pts = [(x + shift_x, y + shift_y) for (x, y) in pts]

    for finger in finger_indices:
        for i in range(len(finger) - 1):
            cv2.line(white_img, shifted_pts[finger[i]], shifted_pts[finger[i + 1]], (0, 255, 0), 2)

    for a, b in palm_connections:
        cv2.line(white_img, shifted_pts[a], shifted_pts[b], (0, 255, 0), 2)

    for point in shifted_pts:
        cv2.circle(white_img, point, 3, (0, 0, 255), -1)

def is_up(tip, pip):
    return tip.y < pip.y

def detect_gesture(left, right):
    gesture = None

    def get_finger_states(hand):
        tips = [hand.landmark[i] for i in [4, 8, 12, 16, 20]]
        pips = [hand.landmark[i] for i in [3, 6, 10, 14, 18]]
        return [is_up(tips[i], pips[i]) for i in range(5)]

    # Detect basic one-handed static gestures
    for hand in [left, right]:
        if hand:
            states = get_finger_states(hand)
            if all(states):
                return "Hello"
            if states[1] and states[2] and not states[3] and not states[4]:
                return "Peace"
            if states[0] and states[1] and not states[2] and not states[3] and states[4]:
                return "I Love You"

    # Two-handed: More
    if left and right:
        l_index_tip = left.landmark[8]
        r_index_tip = right.landmark[8]
        l_thumb_tip = left.landmark[4]
        r_thumb_tip = right.landmark[4]

        dist_thumb = np.linalg.norm(np.array([l_thumb_tip.x - r_thumb_tip.x, l_thumb_tip.y - r_thumb_tip.y]))
        dist_index = np.linalg.norm(np.array([l_index_tip.x - r_index_tip.x, l_index_tip.y - r_index_tip.y]))

        if dist_thumb < 0.1 and dist_index < 0.1:
            return "More"

    # YES (one fist doing up/down motion)
    if right:
        states = get_finger_states(right)
        if not any(states):  # Fist
            wrist_y = right.landmark[0].y
            right_wrist_history.append(wrist_y)
            if len(right_wrist_history) >= 5:
                delta = max(right_wrist_history) - min(right_wrist_history)
                if delta > 0.05:
                    return "Yes"

    return gesture

# Webcam capture
cap = cv2.VideoCapture(0)
gesture_sentence = []

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        white_img = np.ones((400, 800, 3), np.uint8) * 255

        if results.left_hand_landmarks:
            draw_hand_skeleton(results.left_hand_landmarks, white_img, (600, 200))
            cv2.putText(white_img, "Left", (580, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if results.right_hand_landmarks:
            draw_hand_skeleton(results.right_hand_landmarks, white_img, (200, 200))
            cv2.putText(white_img, "Right", (180, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        gesture_text = detect_gesture(results.left_hand_landmarks, results.right_hand_landmarks)

        if gesture_text and (len(gesture_sentence) == 0 or gesture_text != gesture_sentence[-1]):
            gesture_sentence.append(gesture_text)
            if len(gesture_sentence) > 6:
                gesture_sentence.pop(0)

        if gesture_text:
            cv2.putText(white_img, f"Gesture: {gesture_text}", (280, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

        sentence = " ".join(gesture_sentence)
        cv2.putText(white_img, f"Sentence: {sentence}", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 150), 2)

        cv2.imshow("Webcam Feed", frame)
        cv2.imshow("Hand Skeleton Simulator", white_img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
