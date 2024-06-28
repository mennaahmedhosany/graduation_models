from flask import Flask, render_template, request
import pickle
import cv2
import string
import language_tool_python
import mediapipe as mp
from flask_socketio import SocketIO, emit
from threading import Lock
import numpy as np
import speech_recognition as sr
from my_functions import image_process, draw_landmarks, keypoint_extraction

actions = ['name', 'a', 'e', 'hello', 'i love you', 'is', 'm', 'my', 'n', 'no', 'thank you',
           'what', 'yes', 'your']

# Create Flask app and SocketIO
flask_app = Flask(__name__)
flask_app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(flask_app, async_mode='threading')

# Load the machine learning model
model = pickle.load(open("model.pkl", "rb"))

# Create an instance of the grammar correction tool
tool = language_tool_python.LanguageToolPublicAPI('en-UK')

# Initialize global variables and locks
sentence, keypoints, last_prediction, grammar_result = [], [], None, None
video_thread, audio_thread = None, None
thread_lock = Lock()
running = False
capturing = False

# Function to capture video from webcam and process it
def capture_and_process_video():
    global sentence, keypoints, last_prediction, grammar_result, capturing

    cap = cv2.VideoCapture(0)  # Initialize the webcam
    if not cap.isOpened():
        return "Error: Could not open video capture."

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        while cap.isOpened() and capturing:
            ret, image = cap.read()
            if not ret:
                break

            results = image_process(image, holistic)
            draw_landmarks(image, results)
            keypoints.append(keypoint_extraction(results))

            # Process keypoints every 10 frames
            if len(keypoints) == 10:
                keypoints_array = np.array(keypoints).reshape(1, 10, -1)  # Ensure the shape matches model input
                prediction = model.predict(keypoints_array)
                keypoints = []

                # Update sentence based on prediction
                if np.amax(prediction) > 0.9:
                    if last_prediction != actions[np.argmax(prediction)]:
                        sentence.append(actions[np.argmax(prediction)])
                        last_prediction = actions[np.argmax(prediction)]

            # Keep only the last 7 sentences
            if len(sentence) > 7:
                sentence = sentence[-7:]

            # Capitalize the first word
            if sentence:
                sentence[0] = sentence[0].capitalize()

            # Merge words if necessary
            if len(sentence) >= 2:
                if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                    if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                        sentence[-1] = sentence[-2] + sentence[-1]
                        sentence.pop(len(sentence) - 2)
                        sentence[-1] = sentence[-1].capitalize()

            # When sentence length reaches 10, correct grammar
            if len(sentence) == 10:
                text = ' '.join(sentence)
                grammar_result = tool.correct(text)
                sentence = []

            # Determine text to display
            if grammar_result:
                text_to_display = grammar_result
            else:
                text_to_display = ' '.join(sentence)

            # Emit caption update to clients for video
            socketio.emit('video_caption_update', {'caption': text_to_display})

            # Display text on image
            image.setflags(write=1)
            textsize = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2
            cv2.putText(image, text_to_display, (text_X_coord, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Camera', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return grammar_result if grammar_result else ' '.join(sentence)

# Function to transcribe audio from the microphone
def transcribe_audio():
    global running
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)  # Adjust for noise
        while running:
            print("Listening for audio...")
            audio = recognizer.listen(source)  # Listen to the microphone input
            try:
                text = recognizer.recognize_google(audio)  # Use Google Speech Recognition
                print(f"Recognized audio: {text}")
                socketio.emit('audio_caption_update', {'caption': text})
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

# Route to render the index.html template
@flask_app.route("/")
def home():
    return render_template("home.html")

# Route to start captioning process
@flask_app.route("/start_captioning", methods=["POST"])
def start_captioning():
    global video_thread, capturing
    capturing = True
    with thread_lock:
        if video_thread is None:
            video_thread = socketio.start_background_task(target=capture_and_process_video)
    return render_template("home.html")

# Route to stop captioning process
@flask_app.route("/stop_captioning", methods=["POST"])
def stop_captioning():
    global capturing
    capturing = False
    return render_template("home.html")

# Route to start audio transcription process
@flask_app.route("/start_audio_transcription", methods=["POST"])
def start_audio_transcription():
    global running, audio_thread
    running = True
    with thread_lock:
        if audio_thread is None:
            audio_thread = socketio.start_background_task(target=transcribe_audio)
    return render_template("home.html")

# Route to stop audio transcription process
@flask_app.route("/stop_audio_transcription", methods=["POST"])
def stop_audio_transcription():
    global running
    running = False
    return render_template("home.html")

# Main entry point
if __name__ == "__main__":
    socketio.run(flask_app, debug=True, host='0.0.0.0', port=5000, use_reloader=True, allow_unsafe_werkzeug=True)
