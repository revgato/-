import cv2
import time
from classes.computer import Computer
from tensorflow.keras.models import load_model
import mediapipe as mp
from classes.user import User
import copy
import numpy as np
import os


# Load model
model_path = os.path.join(os.getcwd(), "model/weights.hdf5")
model = load_model(model_path)

# Initalize mediapipe hand-landmark detection

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 1,
            model_complexity=0, min_detection_confidence=0.5,
            min_tracking_confidence=0.5) 

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


# Define labels
labels = ["None", "Rock", "Paper", "Scissors", "Like", "Dislike"]

# Load label's image and resize it
dim = (30, 50)

rock_img = cv2.imread(os.path.join(os.getcwd(), "classes/rock.png"))
rock_img = cv2.resize(rock_img, dim, interpolation = cv2.INTER_AREA)

paper_img = cv2.imread(os.path.join(os.getcwd(), "classes/paper.png"))
paper_img = cv2.resize(paper_img, dim, interpolation = cv2.INTER_AREA)

scissors_img = cv2.imread(os.path.join(os.getcwd(), "classes/scissors.png"))
scissors_img = cv2.resize(scissors_img, dim, interpolation = cv2.INTER_AREA)


class Controller:
    # This class helps us control the basic operations of the game
    def __init__(self, cap):
        self.cap = cap
        self.user = User()
        self.computer = Computer()

        self.labels_img = {
            "Rock" : rock_img,
            "Paper": paper_img,
            "Scissors" : scissors_img
        }


    
    def welcome(self, count_down):
        # This function displays the sentences "Welcome!" welcome 
        # on the display for "count_down" second

        tic = time.time()
        toc = time.time()
        countdown = toc - tic

        while(countdown < count_down):
            ret, frame = self.cap.read()
            if ret:
                # Flip the image horizontally for a selfie-view display
                frame = cv2.flip(frame, 1)

                frame = self.putText(frame, "Welcome! ",  (150, 200), 2, 4)
                toc = time.time()
                countdown = toc - tic
                cv2.imshow("Game", frame)
                cv2.waitKey(1)


    def ask(self):
        # This function will ask the player if he wants to continue playing. 
        # Player gestures "Like" to continue playing, "Dislike" to exit
        
        while True:
            ret, frame = self.cap.read()
            if ret:
                # Flip the image horizontally for a selfie-view display
                frame = cv2.flip(frame, 1)

                frame  = self.show_results(frame)

                frame = self.putText(frame, "Like to continue", (100, 200), 1.5, 4)
                frame = self.putText(frame, "Dislike to exit", (100, 300), 1.5, 4)

                predict = self.waitSignal(frame)

                cv2.imshow("Game", frame)
                cv2.waitKey(1)

                if predict == "Dislike":
                    return "Exit"
                elif predict == "Like":
                    return "Continue"


    def game_play(self, count_down):
        # This function wait for the user's gesture. 
        # Then calculate the result and display it on the screen

        tic = time.time()
        toc = time.time()
        countdown = toc - tic

        # Initialize Computer's signal
        self.computer.makeSignal(labels)

        # Set current winner = None
        winner = ""

        # If the user holds their gesture for TIME_HOLD_SIGNAL seconds, record the result and proceed with the calculation.
        while(countdown < count_down):
            ret, frame = self.cap.read()
            if ret:
                # Flip the image horizontally for a selfie-view display
                frame = cv2.flip(frame, 1)
                frame = self.show_results(frame)
                self.user.signal = self.waitSignal(frame)

                if self.user.signal not in labels[1:4]:
                    # While user's signal is not Rock, Paper or Scissors, then 
                    # count down again for TIME_HOLD_SIGNAL second
                    tic = time.time()
                
                else:
                    # If the user outputs a Rock, Paper or Scissors signal
                    frame = self.showSignal(frame)
                    winner = self.winner()
                    frame = self.putText(frame, winner, (10, 300), 1.5, 4)
                
                toc = time.time()
                countdown = toc - tic
                cv2.imshow("Game", frame)
                cv2.waitKey(1)

        # Calculate the score
        self.score_calculation(winner)



    def countdown(self, count_down):

        # This function will countdown within TIME_PER_ROUND seconds, 
        # and displaying on the screen the number of seconds remaining count down
        tic = time.time()
        toc = time.time()
        countdown = 0

        for second in range(count_down):
            while True:
                ret, frame = self.cap.read()
                if ret:
                    # Flip the image horizontally for a selfie-view display
                    frame = cv2.flip(frame, 1)
                    frame = self.show_results(frame)
                    frame = self.putText(frame, str(count_down - second), (200, 400), 13, 8)
                    
                    toc = time.time()
                    countdown = toc - tic
                    cv2.imshow("Game", frame)
                    cv2.waitKey(1) 

                    if countdown > second:
                        break

    def showSignal(self, frame):
        # This function helps us to display the labels as an image 
        # when the user and the computer make a decision

        frame = self.putText(frame, "You: ", (10, 400), 1, 1)
        frame = self.putText(frame, "Computer: ", (10, 450), 1, 1)
        frame[370:370 + dim[1], 180: 180 + dim[0]] = self.labels_img[self.user.signal] 
        frame[420:420 + dim[1], 180: 180 + dim[0]] = self.labels_img[self.computer.signal] 

        return frame


    def waitSignal(self, frame):
        # This function uses Mediapipe to determine the hand landmark and then recognizes 
        # the gesture through the model. If hand landmark is not found, return None label
        frame.flags.writeable=True
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        frame.flags.writeable=True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                hand_landmarks_calc = calc_landmark_list(frame, 
                        copy.deepcopy(hand_landmarks))
                
                predict = model.predict(hand_landmarks_calc).argmax(axis=1)

                return labels[predict[0]]
        return "None"


    def winner(self):
        # This function determines the winner using the law of rock-paper-scissors
        # and returns a message
        computer_signal = self.computer.signal
        user_signal = self.user.signal

        if computer_signal == user_signal:
            return "DRAW!"

        elif computer_signal == "Rock" and user_signal == "Paper":
            return "Congratulations, you win!!!"

        elif computer_signal == "Paper" and user_signal == "Scissors":
            return "Congratulations, you win!!!"

        elif computer_signal == "Scissors" and user_signal == "Rock":
            return "Congratulations, you win!!!"

        else:
            return "You lose :(("


    def score_calculation(self, winner):
        # This function receives the message from the "winner" function 
        # and adds points to the winner
        # Ignore if DRAW

        if winner == "Congratulations, you win!!!":
            self.user.score += 1
        elif winner == "You lose :((":
            self.computer.score += 1
        
        

    def putText(self, frame, text, position, size, thin):
        # This function helps to display text on the screen in a simple way
        return cv2.putText(frame, text, 
                    (position[0], position[1]), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 0), thin, cv2.LINE_AA)

    def show_results(self, frame):
        # This function displays the current score of the user and computer
        frame = self.putText(frame, "Computer Score: {}".format(self.computer.score), (10, 30), 1, 1)

        frame = self.putText(frame, "Your Score: {}".format(self.user.score), (10, 60), 1, 1)
        return frame


