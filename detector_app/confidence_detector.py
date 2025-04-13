import cv2
import mediapipe as mp
import numpy as np

def run_confidence_detection():
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    face_mesh = mp_face_mesh.FaceMesh()
    pose = mp_pose.Pose()
    hands = mp_hands.Hands()

    cap = cv2.VideoCapture(0)

    HEAD_TILT_THRESHOLD = 15
    HAND_MOVEMENT_THRESHOLD = 20
    prev_hand_landmarks = None

    def calculate_head_tilt(face_landmarks, image_width, image_height):
        left_eye = np.array([face_landmarks[33].x * image_width, face_landmarks[33].y * image_height])
        right_eye = np.array([face_landmarks[263].x * image_width, face_landmarks[263].y * image_height])
        dx, dy = right_eye - left_eye
        angle = np.degrees(np.arctan2(dy, dx))
        return abs(angle)

    def detect_confidence(frame):
        nonlocal prev_hand_landmarks
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(image_rgb)
        pose_results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)

        image_height, image_width, _ = frame.shape
        confidence_score = 0

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                head_tilt = calculate_head_tilt(face_landmarks.landmark, image_width, image_height)
                confidence_score += 1 if head_tilt < HEAD_TILT_THRESHOLD else -1

        if pose_results.pose_landmarks:
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

            shoulder_y_diff = abs(left_shoulder.y - right_shoulder.y) * image_height
            hip_y_diff = abs(left_hip.y - right_hip.y) * image_height

            confidence_score += 1 if shoulder_y_diff < 10 and hip_y_diff < 10 else -1

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                current_hand_landmarks = np.array([(lm.x * image_width, lm.y * image_height) for lm in hand_landmarks.landmark])
                if prev_hand_landmarks is not None:
                    movement = np.linalg.norm(current_hand_landmarks - prev_hand_landmarks)
                    confidence_score += 1 if movement < HAND_MOVEMENT_THRESHOLD else -1
                prev_hand_landmarks = current_hand_landmarks

        if confidence_score >= 2:
            return "Confident", (0, 255, 0)
        elif confidence_score <= -2:
            return "Under-confident", (0, 0, 255)
        else:
            return "Neutral", (255, 255, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        status, color = detect_confidence(frame)
        cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Confidence Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
