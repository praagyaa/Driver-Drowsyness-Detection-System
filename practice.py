import streamlit as st
import cv2
from ultralytics import YOLO
import time
import threading
import pygame

# Initialize pygame mixer for sound alerts
pygame.mixer.init()

# Load the YOLO model
model = YOLO("best.pt")

# Control threading for beep sound
continuous_beep_thread = None
stop_continuous_beep = threading.Event()

def play_single_beep():
    """Plays a single beep sound."""
    try:
        sound = pygame.mixer.Sound("beep.mp3")
        sound.play()
        time.sleep(4)  # Match beep duration
    except Exception as e:
        print(f"Error playing sound: {e}")

def play_continuous_beep_for_duration():
    """Plays a continuous beep for 15 seconds if drowsiness is detected."""
    start_time = time.time()
    stop_continuous_beep.clear()
    while not stop_continuous_beep.is_set() and (time.time() - start_time) < 15:
        play_single_beep()

def detect_drowsiness_stream(video_source=0):
    """Detects drowsiness using the YOLO model and webcam feed."""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Error: Could not open video source.")
        return

    stframe = st.empty()
    status_text = st.empty()

    # Manage session state for stopping detection and beep
    if "stop_detection" not in st.session_state:
        st.session_state.stop_detection = False
    if "stop_beep" not in st.session_state:
        st.session_state.stop_beep = False

    eyes_closed_start = None  # Track when eyes first closed
    continuous_beep_active = False

    # Create buttons for stopping detection and beep
    col1, col2 = st.columns(2)
    with col1:
        stop_button = st.button("Stop Detection")
    with col2:
        stop_beep_button = st.button("Stop Beep")

    if stop_button:
        st.session_state.stop_detection = True
    if stop_beep_button:
        stop_continuous_beep.set()
        st.session_state.stop_beep = False

    while not st.session_state.stop_detection:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture frame.")
            break

        results = model(frame)
        current_time = time.time()
        drowsy_detected = False
        eyes_closed_detected = False

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if model.names[class_id] == "Drowsy":
                    eyes_closed_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf[0]

                    # Calculate duration of eye closure
                    if eyes_closed_start is None:
                        eyes_closed_start = current_time
                    eyes_closed_duration = current_time - eyes_closed_start

                    # Drowsy if eyes closed for more than 3 seconds
                    if eyes_closed_duration >= 3.0:
                        drowsy_detected = True
                        color = (0, 0, 255)  # Red for drowsy
                        status = "DROWSY"
                    else:
                        color = (0, 255, 0)  # Green for alert
                        status = "Eyes Closed"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{status} {confidence:.2f} ({eyes_closed_duration:.1f}s)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"Eyes Closed: {eyes_closed_duration:.1f}s",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Reset eyes_closed_start if eyes are open
        if not eyes_closed_detected:
            eyes_closed_start = None
            status_text.success("Alert - Eyes Open")
        elif eyes_closed_detected and not drowsy_detected:
            status_text.warning(f"Eyes Closed for {(current_time - eyes_closed_start):.1f}s")
        else:
            status_text.error("DROWSY ALERT!")

        # Handle continuous beep activation
        if drowsy_detected and not continuous_beep_active:
            continuous_beep_active = True
            global continuous_beep_thread
            if continuous_beep_thread is None or not continuous_beep_thread.is_alive():
                continuous_beep_thread = threading.Thread(
                    target=play_continuous_beep_for_duration,
                    daemon=True
                )
                continuous_beep_thread.start()
        elif not drowsy_detected:
            continuous_beep_active = False

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    stop_continuous_beep.set()
    st.success("Detection stopped.")

st.title("Drowsiness Detection Web App")

if st.button("Start Detection"):
    detect_drowsiness_stream()
