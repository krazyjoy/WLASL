import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Initialize webcam
capture = cv2.VideoCapture(0)

# Initialize hand detector with confidence level
hd = HandDetector(detectionCon=0.7, maxHands=2)
hd2 = HandDetector(detectionCon=0.7, maxHands=2)

offset = 30
white = np.ones((400, 800, 3), np.uint8) * 255  # Expanded width to 800 for two hands

while True:
    try:
        success, frame = capture.read()
        if not success or frame is None:
            print("Warning: Failed to read from camera.")
            continue  # Skip this frame if camera feed fails

        frame = cv2.flip(frame, 1)

        # Correctly extracting hands and processed frame
        hands, frame = hd.findHands(frame, draw=True, flipType=True)

        if hands:
            white_img = np.ones((400, 800, 3), np.uint8) * 255  # Side-by-side display
            hand_positions = [(50, 50), (450, 50)]  # Two positions for left and right hand

            for i, hand in enumerate(hands):
                x, y, w, h = hand['bbox']

                # Ensure bounding box is within frame dimensions
                if x - offset < 0 or y - offset < 0 or x + w + offset > frame.shape[1] or y + h + offset > frame.shape[
                    0]:
                    print(f"Skipping hand {i}: Bounding box out of frame bounds.")
                    continue

                image = frame[y - offset:y + h + offset, x - offset:x + w + offset]

                # Ensure the cropped image is not empty before processing
                if image.size == 0:
                    print(f"Skipping hand {i}: Cropped image is empty.")
                    continue

                handz, _ = hd2.findHands(image, draw=False, flipType=True)
                if handz:
                    for hand in handz:
                        if 'lmList' not in hand:
                            print(f"Skipping hand {i}: Landmark list missing.")
                            continue  # Skip if no landmarks found

                        pts = hand['lmList']
                        os_x, os_y = hand_positions[i % 2]  # Assign position based on index

                        # Draw skeleton
                        for t in range(0, 4, 1):
                            if t + 1 < len(pts):
                                cv2.line(white_img, (pts[t][0] + os_x, pts[t][1] + os_y),
                                         (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        for t in range(5, 8, 1):
                            if t + 1 < len(pts):
                                cv2.line(white_img, (pts[t][0] + os_x, pts[t][1] + os_y),
                                         (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        for t in range(9, 12, 1):
                            if t + 1 < len(pts):
                                cv2.line(white_img, (pts[t][0] + os_x, pts[t][1] + os_y),
                                         (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        for t in range(13, 16, 1):
                            if t + 1 < len(pts):
                                cv2.line(white_img, (pts[t][0] + os_x, pts[t][1] + os_y),
                                         (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        for t in range(17, 20, 1):
                            if t + 1 < len(pts):
                                cv2.line(white_img, (pts[t][0] + os_x, pts[t][1] + os_y),
                                         (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)

                        # Connect key joints
                        cv2.line(white_img, (pts[5][0] + os_x, pts[5][1] + os_y), (pts[9][0] + os_x, pts[9][1] + os_y),
                                 (0, 255, 0), 3)
                        cv2.line(white_img, (pts[9][0] + os_x, pts[9][1] + os_y),
                                 (pts[13][0] + os_x, pts[13][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white_img, (pts[13][0] + os_x, pts[13][1] + os_y),
                                 (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white_img, (pts[0][0] + os_x, pts[0][1] + os_y), (pts[5][0] + os_x, pts[5][1] + os_y),
                                 (0, 255, 0), 3)
                        cv2.line(white_img, (pts[0][0] + os_x, pts[0][1] + os_y),
                                 (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)

                        # Draw joint points
                        for j in range(len(pts)):
                            cv2.circle(white_img, (pts[j][0] + os_x, pts[j][1] + os_y), 2, (0, 0, 255), 1)

                    cv2.imshow("Skeleton", white_img)

        cv2.imshow("Hand Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    except Exception as e:
        print("Error:", e)
        continue  # Keep the program running instead of crashing

capture.release()
cv2.destroyAllWindows()