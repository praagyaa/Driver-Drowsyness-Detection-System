from flask import Flask, render_template, redirect, url_for, request, session, flash, Response
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import threading
import time
import cv2
import pygame
from ultralytics import YOLO

app = Flask(__name__)


# Secret key for session management
app.secret_key = 'your_secret_key'

# MySQL database configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  # MySQL username
app.config['MYSQL_PASSWORD'] = ''  # Leave empty if no password
app.config['MYSQL_DB'] = 'drowsiness_monitor'  # Database name

# Initialize MySQL
mysql = MySQL(app)

# Initialize pygame for sound
pygame.mixer.init()

# Load YOLO model
model = YOLO("best.pt")

# Threading control for beep and detection
detection_thread = None
detection_active = False  # To track if detection is active


def play_single_beep():
    try:
        sound = pygame.mixer.Sound("beep.mp3")
        sound.play()
        time.sleep(4)
    except Exception as e:
        print(f"Error playing sound: {e}")

def play_continuous_beep_for_duration():
    start_time = time.time()
    while time.time() - start_time < 15:
        play_single_beep()

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    license_number = session.get('license')
    cursor.execute("SELECT * FROM users WHERE license_number = %s", (license_number,))
    account = cursor.fetchone()
    name = account["name"]
    
    eyes_closed_start = None
    while detection_active:
        ret, frame = cap.read()
        if not ret:
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
                    
                    if eyes_closed_start is None:
                        eyes_closed_start = current_time
                    
                    eyes_closed_duration = current_time - eyes_closed_start
                    
                    if eyes_closed_duration >= 3.0:
                        drowsy_detected = True
                        color = (0, 0, 255)  # Red for drowsy
                        label = f"Drowsy {confidence:.2f} ({eyes_closed_duration:.1f}s)"
                    else:
                        color = (0, 255, 0)  # Green for alert
                        label = f"Eyes Closed ({eyes_closed_duration:.1f}s)"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if not eyes_closed_detected:
            eyes_closed_start = None
        
        if drowsy_detected:
            global detection_thread
            if detection_thread is None or not detection_thread.is_alive():
                detection_thread = threading.Thread(target=play_continuous_beep_for_duration, daemon=True)
                detection_thread.start()
                cursor.execute("INSERT INTO users (name, license_number) VALUES (%s, %s)", (name, license_number))


        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
@app.route('/frontend')
def frontend():
    return render_template('frontend.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['username'].strip()
        license_number = request.form['license'].strip()
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM users WHERE license_number = %s", (license_number,))
        account = cursor.fetchone()
        
        # License number validation (XX-XX-XXXXXXXX)
        license_pattern = r'^\d{2}-\d{2}-\d{8}$'

        if account:
            flash("License number already registered! Please login.", "danger")
            return render_template('message.html')
            #return redirect(url_for('login'))  # Redirect to login if already signed up
        elif not re.match(r'^[a-zA-Z\s]+$', name):
            flash("Name must contain only letters and spaces.", "danger")
        elif not re.match(license_pattern, license_number):
            flash("License number must be in the format XX-XX-XXXXXXXX (e.g., 12-34-56789012).", "danger")
        else:
            cursor.execute("INSERT INTO users (name, license_number) VALUES (%s, %s)", (name, license_number))
            mysql.connection.commit()
            cursor.close()
            session['name'] = name
            flash(f"Registration successful! Welcome, {name}.", "success")
            #return redirect(url_for('drowsiness_detection'))  # Redirect to detection page
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        license_number = request.form['license'].strip()
        
        # License number validation (XX-XX-XXXXXXXX)
        license_pattern = r'^\d{2}-\d{2}-\d{8}$'

        if not re.match(license_pattern, license_number):
            flash("License number must be in the format XX-XX-XXXXXXXX (e.g., 12-34-56789012).", "danger")
            return redirect(url_for('login'))  # Prevents unnecessary DB queries
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute("SELECT * FROM users WHERE license_number = %s", (license_number,))
        account = cursor.fetchone()
        
        if account:
            session['license'] = account['license_number']
            flash(f"Welcome back, {account['name']}!", "success")
            return redirect(url_for('drowsiness_detection'))
            #return redirect(url_for('drowsiness_detection', name=account['name']))
        else:
            flash("License number not registered. Please sign up first.", "danger")
            return redirect(url_for('signup'))  # Redirect to signup page
    
    return render_template('login.html')

@app.route('/drowsiness_detection')
def drowsiness_detection():
    if 'license' not in session:
        return redirect(url_for('login'))
    name = request.args.get('name')
    return render_template('drowsiness_detection.html')
    #return render_template('drowsiness_detection.html', name=name)


@app.route('/logout')
def logout():
    session.pop('name', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))

@app.route('/start-detection', methods=['POST'])
def start_detection():
    global detection_active
    detection_active = True
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    license_number = session.get('license')
    cursor.execute("SELECT * FROM users WHERE license_number = %s", (license_number,))
    account = cursor.fetchone()
    name = account["name"]
    return redirect(url_for('drowsiness_detection'))
    #return redirect(url_for('drowsiness_detection', name=name))

@app.route('/stop-detection', methods=['POST'])
def stop_detection():
    global detection_active
    detection_active = False 
     # Stop detection
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    license_number = session.get('license')
    cursor.execute("SELECT * FROM users WHERE license_number = %s", (license_number,))
    account = cursor.fetchone()
    name = account["name"]
    return redirect(url_for('drowsiness_detection'))
    #return redirect(url_for('drowsiness_detection', name=name))

@app.route('/')
@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)