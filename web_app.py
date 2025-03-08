import streamlit as st
import cv2
from ultralytics import YOLO
import time
import threading
from playsound import playsound


model = YOLO("best.pt")  


continuous_beep_thread = None
stop_continuous_beep = threading.Event()

def play_single_beep():
    playsound('beep.mp3') 

def play_continuous_beep_for_duration():

    start_time = time.time()
    stop_continuous_beep.clear()
    while not stop_continuous_beep.is_set() and (time.time() - start_time) < 15:
        playsound('beep.mp3')
        time.sleep(0.1) 

def detect_drowsiness_stream(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("Error: Could not open video source.")
        return

    stframe = st.empty()


    if "stop_detection" not in st.session_state:
        st.session_state.stop_detection = False
    if "stop_beep" not in st.session_state:
        st.session_state.stop_beep = False


    drowsy_events = [] 
    last_single_beep_time = 0
    continuous_beep_active = False

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

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if model.names[class_id] == "Drowsy":
                    drowsy_detected = True
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Drowsy {confidence:.2f}", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if drowsy_detected:
            drowsy_events.append(current_time)
            
    
            drowsy_events = [t for t in drowsy_events if current_time - t <= 60]

            if current_time - last_single_beep_time >= 1.0:  
                threading.Thread(target=play_single_beep, daemon=True).start()
                last_single_beep_time = current_time

            if len(drowsy_events) >= 6 and not continuous_beep_active:
                continuous_beep_active = True
                global continuous_beep_thread
                if continuous_beep_thread is None or not continuous_beep_thread.is_alive():
                    continuous_beep_thread = threading.Thread(
                        target=play_continuous_beep_for_duration, 
                        daemon=True
                    )
                    continuous_beep_thread.start()

        if continuous_beep_active and (continuous_beep_thread is None or not continuous_beep_thread.is_alive()):
            continuous_beep_active = False

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    stop_continuous_beep.set()
    st.success("Detection stopped.")

# Streamlit app layout
st.title("Drowsiness Detection Web App")

if st.button("Start Detection"):
    detect_drowsiness_stream()