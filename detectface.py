import cv2
import numpy as np
from keras.models import load_model
from collections import Counter
from keras.applications.vgg16 import preprocess_input

# Load model globally once
model = load_model("facial emotion detection\\vgg16+transformer.keras")
emotion_labels = ['angry', 'happy', 'sad', 'surprise', 'neutral']
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Same preprocessing as training: RGB, 128x128, VGG16 preprocess_input
def predict_face(face_crop):
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (128,128)).astype("float32")
    face_input = preprocess_input(np.expand_dims(face_resized, axis=0))
    pred = model.predict(face_input, verbose=0)
    return emotion_labels[np.argmax(pred)]

def predict_emotion_image(img_path):
    image = cv2.imread(img_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detectMultiScale(rgb, 1.1, 5)
    for (x, y, w, h) in faces:
        face_crop = image[y:y+h, x:x+w]
        return predict_face(face_crop)
    return "No Face Detected"

def predict_emotion_webcam():
    cap = cv2.VideoCapture(0)
    emotions_detected = []
    total_frames = 60
    frame_count = 0

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detectMultiScale(rgb, 1.1, 5)
        display = frame.copy()
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            emotion = predict_face(face_crop)
            emotions_detected.append(emotion)
            # Draw the cascade box and predicted emotion on the live preview
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite("static/uploads/temp.jpg", display)

        cv2.imshow("MoodMate - Webcam (press q to stop)", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    if emotions_detected:
        majority_emotion = Counter(emotions_detected).most_common(1)[0][0]
        return majority_emotion
    return "No Face Detected"
