import cv2
import dlib
from keras.models import load_model
import numpy as np

# Load pre-trained face detector and expression recognition model
face_detector = dlib.get_frontal_face_detector()
expression_model = load_model('path/to/your/expression/model.h5')

# Function to preprocess face image for expression classification
def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = np.expand_dims(face, axis=0)
    face = face / 255.0  # Normalize pixel values
    return face

# Load video file
video_path = '.\m.imaduddin.ar@gmail.com_converter.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray_frame)

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        # Extract the region of interest (ROI) containing the face
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess the face for expression classification
        processed_face = preprocess_face(face_roi)

        # Predict the expression using the trained model
        expression_probabilities = expression_model.predict(processed_face)
        predicted_expression = np.argmax(expression_probabilities)

        # Display the predicted expression on the frame
        expression_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        expression_text = expression_labels[predicted_expression]
        cv2.putText(frame, expression_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with the detected faces and expressions
    cv2.imshow('Microexpression Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()