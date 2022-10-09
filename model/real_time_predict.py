import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import copy
import os
from tensorflow.keras.models import load_model


# This file helps us try to real-time predict hand gestures. Press ESC to exit.

# Load model
model = load_model(os.path.join(os.getcwd(), "model/weights.hdf5"))


# This function helps convert keypoint of hand landmarks 
# to relative coordinates and normalize, flatten it
def calc_landmark_list(frame, hand_landmarks):
    height, width, _ = frame.shape

    landmark_list = []     # Keypoint

    for landmark in hand_landmarks.landmark:
        # It may happen that x > 1 (mediapipe estimate), we need to handle it
        x = min(int(landmark.x * width), width - 1)
        y = min(int(landmark.y * height), height - 1)
        # Ignore z

        landmark_list.append([x, y])
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        
        landmark_point[0] = landmark_point[0] - base_x
        landmark_point[1] = landmark_point[1] - base_y

    # Convert to 1 dimensional array
    landmark_list = np.reshape(landmark_list, (42, 1))

    # Normalization
    max_value = max(map(abs, landmark_list))
    landmark_list = (landmark_list / max_value).T

    return landmark_list



# This function helps compute the bounding box of detected hand
def calc_bbox(frame, hand_landmarks):
    height, width, _ = frame.shape
    
    landmark_list = []     # Keypoint

    for landmark in hand_landmarks.landmark:
        # It may happen that x > 1 (mediapipe estimate), we need to handle it
        x = min(int(landmark.x * width), width - 1)
        y = min(int(landmark.y * height), height - 1)
        # Ignore z

        landmark_list.append([x, y])

    landmark_list = np.asarray(landmark_list)
    x, y, w, h = cv2.boundingRect(landmark_list)

    return [x, y, x + w, y + h]



# Initialize mediapipe classes
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Define list of Labels
labels = ["None", "Rock", "Paper", "Scissors", "Like", "Dislike"]
label_num = len(labels)
num_labeled = np.zeros(label_num, dtype = "int")

# Read frame from webcam
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands = 1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if ret:
            # To improve performance, optionally mark the frame as not writeable to
            # pass by reference.

            # Flip the image horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)

            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            frame.flags.writeable=True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw the hand annotations on the frame
            if results.multi_hand_landmarks:

                for hand_landmarks in results.multi_hand_landmarks:
                    # hand_lankmarks = list of 21 coordinate points (x-y-z)
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # Convert hand_landmarks to relative coordinates keypoint
                    # and normalize, flatten it
                    hand_landmarks_calc = calc_landmark_list(frame, 
                        copy.deepcopy(hand_landmarks))

                    # Use models to predict hand gestures and display them on screen
                    data = np.expand_dims(hand_landmarks_calc, axis = 0)
                    predict = model.predict(hand_landmarks_calc).argmax(axis=1)
                    cv2.putText(frame, labels[predict[0]], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    # Compute and draw bounding-box
                    x1, y1, x2, y2 = calc_bbox(frame, hand_landmarks)
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                        (0, 0, 0), 1)


             # If user press ESC, then exit program       
            if cv2.waitKey(1) == 27:
                break

        # Full-screen mode
        cv2.namedWindow("Webcam", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Webcam",cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Webcam", frame)
    


cap.release()
cv2.destroyAllWindows()