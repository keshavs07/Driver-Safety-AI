import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading

# ========== INIT SOUND ==========
pygame.mixer.init()

def play_alarm():
    pygame.mixer.music.load("new_alarm.wav")
    pygame.mixer.music.play(-1)  # loop continuously

def stop_alarm():
    pygame.mixer.music.stop()

# ========== MEDIAPIPE ==========
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ========== FUNCTIONS ==========
def calculate_ear(eye_points, landmarks, w, h):
    coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in eye_points]

    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    h1 = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))

    return (v1 + v2) / (2.0 * h1)

def calculate_mar(landmarks, w, h):
    top = (int(landmarks[13].x * w), int(landmarks[13].y * h))
    bottom = (int(landmarks[14].x * w), int(landmarks[14].y * h))
    left = (int(landmarks[78].x * w), int(landmarks[78].y * h))
    right = (int(landmarks[308].x * w), int(landmarks[308].y * h))

    vertical = np.linalg.norm(np.array(top) - np.array(bottom))
    horizontal = np.linalg.norm(np.array(left) - np.array(right))

    return vertical / horizontal

# ========== CAMERA ==========
cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6

EYE_FRAMES = 20
YAWN_FRAMES = 15

eye_counter = 0
yawn_counter = 0
alarm_on = False

# ========== LOOP ==========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # EAR (eyes)
            left_ear = calculate_ear(LEFT_EYE, landmarks, w, h)
            right_ear = calculate_ear(RIGHT_EYE, landmarks, w, h)
            ear = (left_ear + right_ear) / 2.0

            # MAR (yawn)
            mar = calculate_mar(landmarks, w, h)

            # Display values
            cv2.putText(frame, f'EAR: {ear:.2f}', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f'MAR: {mar:.2f}', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            # -------- EYE DROWSINESS --------
            if ear < EAR_THRESHOLD:
                eye_counter += 1
                if eye_counter >= EYE_FRAMES:
                    cv2.putText(frame, "SLEEPING ALERT!", (120, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

                    if not alarm_on:
                        alarm_on = True
                        threading.Thread(target=play_alarm, daemon=True).start()
            else:
                eye_counter = 0
                if alarm_on:
                    stop_alarm()
                alarm_on = False

            # -------- YAWN DETECTION --------
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= YAWN_FRAMES:
                    cv2.putText(frame, "YAWNING!", (180, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)
            else:
                yawn_counter = 0

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()