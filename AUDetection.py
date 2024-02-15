import dlib
import cv2

# Load pre-trained face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('.\shape_predictor_68_face_landmarks.dat')

# Load video file
video_path = '.\m.imaduddin.ar@gmail.com_converter.mp4'
cap = cv2.VideoCapture(video_path)

# Set full screen display
cv2.namedWindow('AU Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('AU Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Define labels for each Action Unit
au_labels = {
    'AU1': 'Inner Brow Raiser',
    'AU2': 'Outer Brow Raiser',
    'AU4': 'Brow Lowerer',
    'AU5': 'Upper Lid Raiser',
    'AU6': 'Cheek Raiser',
    'AU7': 'Lid Tightener',
    'AU9': 'Nose Wrinkler',
    'AU10': 'Upper Lip Raiser',
    'AU12': 'Lip Corner Puller',
    'AU14': 'Dimpler',
    'AU15': 'Lip Corner Depressor',
    'AU17': 'Chin Raiser'
}

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray_frame)

    for face in faces:
        # Detect landmarks in the face
        landmarks = landmark_predictor(gray_frame, face)

        # Define coordinates for all 12 Action Units
        au1_start = (landmarks.part(21).x, landmarks.part(21).y)
        au1_end = (landmarks.part(22).x, landmarks.part(22).y)

        au2_start = (landmarks.part(17).x, landmarks.part(17).y)
        au2_end = (landmarks.part(26).x, landmarks.part(26).y)

        au4_start = (landmarks.part(43).x, landmarks.part(43).y)
        au4_end = (landmarks.part(47).x, landmarks.part(47).y)

        au5_start = (landmarks.part(37).x, landmarks.part(37).y)
        au5_end = (landmarks.part(44).x, landmarks.part(44).y)

        au6_start = (landmarks.part(19).x, landmarks.part(19).y)
        au6_end = (landmarks.part(24).x, landmarks.part(24).y)

        au7_start = (landmarks.part(38).x, landmarks.part(38).y)
        au7_end = (landmarks.part(42).x, landmarks.part(42).y)

        au9_start = (landmarks.part(51).x, landmarks.part(51).y)
        au9_end = (landmarks.part(55).x, landmarks.part(55).y)

        au10_start = (landmarks.part(48).x, landmarks.part(48).y)
        au10_end = (landmarks.part(54).x, landmarks.part(54).y)

        au12_start = (landmarks.part(61).x, landmarks.part(61).y)
        au12_end = (landmarks.part(65).x, landmarks.part(65).y)

          # Draw rectangles around the specified Action Units with labels
        cv2.rectangle(frame, au1_start, au1_end, (0, 255, 0), 1)
        cv2.putText(frame, 'AU1: ' + au_labels['AU1'], (au1_start[0], au1_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.rectangle(frame, au2_start, au2_end, (0, 255, 0), 1)
        cv2.putText(frame, 'AU2: ' + au_labels['AU2'], (au2_start[0], au2_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.rectangle(frame, au4_start, au4_end, (0, 255, 0), 1)
        cv2.putText(frame, 'AU4: ' + au_labels['AU4'], (au4_start[0], au4_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.rectangle(frame, au5_start, au5_end, (0, 255, 0), 1)
        cv2.putText(frame, 'AU5: ' + au_labels['AU5'], (au5_start[0], au5_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.rectangle(frame, au6_start, au6_end, (0, 255, 0), 1)
        cv2.putText(frame, 'AU6: ' + au_labels['AU6'], (au6_start[0], au6_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.rectangle(frame, au7_start, au7_end, (0, 255, 0), 1)
        cv2.putText(frame, 'AU7: ' + au_labels['AU7'], (au7_start[0], au7_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.rectangle(frame, au9_start, au9_end, (0, 255, 0), 1)
        cv2.putText(frame, 'AU9: ' + au_labels['AU9'], (au9_start[0], au9_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.rectangle(frame, au10_start, au10_end, (0, 255, 0), 1)
        cv2.putText(frame, 'AU10: ' + au_labels['AU10'], (au10_start[0], au10_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.rectangle(frame, au12_start, au12_end, (0, 255, 0), 1)
        cv2.putText(frame, 'AU12: ' + au_labels['AU12'], (au12_start[0], au12_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)


    # Display the frame with detected faces and rectangles
    cv2.imshow('AU Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()