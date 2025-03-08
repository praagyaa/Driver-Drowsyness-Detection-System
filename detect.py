import cv2
from ultralytics import YOLO
from PIL import Image

# Load the YOLOv8 model
model = YOLO("best.pt")  # Replace with the path to your trained weights

def detect_drowsiness(video_source=0):
    # Open the video capture (webcam)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("[INFO] Starting detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Perform inference using YOLOv8
        results = model(frame)

        # Check if any "Drowsy" class is detected
        drowsy_detected = False
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # Get the class ID
                if model.names[class_id] == "Drowsy":  # Check for the 'Drowsy' class
                    drowsy_detected = True

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf[0]

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"Drowsy {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

        # Display the frame with detection
        try:
            # Convert OpenCV image (BGR) to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(frame_rgb)
            image_pil.show()
        except Exception as e:
            print(f"Error showing image: {e}")

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection stopped.")

# Run the detection
if __name__ == "__main__":
    detect_drowsiness()
