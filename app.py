import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

# ========== INIT SOUND ==========
pygame.mixer.init()

def play_alarm():
    if not pygame.mixer.music.get_busy():
        try:
            pygame.mixer.music.load("new_alarm.wav")
            pygame.mixer.music.play(-1)  # loop continuously
        except Exception:
            pass

def stop_alarm():
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass

# ========== MEDIAPIPE ==========
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

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
camera = None
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
EYE_FRAMES = 20
YAWN_FRAMES = 15

def init_camera(source=0):
    global camera
    if camera is not None:
        camera.release()
    
    try:
        # Handle string numbers like "0" or "1"
        source = int(source)
    except ValueError:
        pass # It is a string URL for an IP camera
        
    camera = cv2.VideoCapture(source)

def generate_frames():
    global camera

    if camera is None:
        init_camera(0)

    yawn_counter = 0
    alarm_on = False
    drowsy_start_time = None

    try:
        while True:
            if camera is None:
                break
                
            ret, frame = camera.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark

                    left_ear = calculate_ear(LEFT_EYE, landmarks, w, h)
                    right_ear = calculate_ear(RIGHT_EYE, landmarks, w, h)
                    ear = (left_ear + right_ear) / 2.0
                    mar = calculate_mar(landmarks, w, h)

                    cv2.putText(frame, f'EAR: {ear:.2f}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f'MAR: {mar:.2f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # Drowsiness and yawn detection logic
                    if ear < EAR_THRESHOLD or mar > MAR_THRESHOLD:
                        if ear < EAR_THRESHOLD:
                            if drowsy_start_time is None:
                                drowsy_start_time = time.time()
                            elapsed_time = time.time() - drowsy_start_time

                            if elapsed_time >= 5:  # Timer set to 5 seconds
                                cv2.putText(frame, "DROWSINESS ALERT!", (120, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                                if not alarm_on:
                                    alarm_on = True
                                    threading.Thread(target=play_alarm, daemon=True).start()
                        else:
                            drowsy_start_time = None # Reset eye timer if eyes are open but yawning

                        if mar > MAR_THRESHOLD:
                            yawn_counter += 1
                            if yawn_counter >= YAWN_FRAMES:
                                cv2.putText(frame, "YAWNING!", (180, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                                if not alarm_on:
                                    alarm_on = True
                                    threading.Thread(target=play_alarm, daemon=True).start()
                        else:
                            yawn_counter = 0
                    else:
                        drowsy_start_time = None
                        yawn_counter = 0
                        if alarm_on:
                            stop_alarm()
                            alarm_on = False

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        if alarm_on:
            stop_alarm()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_camera', methods=['POST'])
def set_camera():
    data = request.json
    source = data.get('source', 0)
    init_camera(source)
    return jsonify({"status": "success", "source": source})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)