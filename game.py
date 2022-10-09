from classes.monitor import Controller
import cv2

TIME_WELCOME = 3    # Welcome time = 3 second
TIME_PER_ROUND = 4  # Countdown between turns = 4 seconds
TIME_HOLD_SIGNAL = 2

cap = cv2.VideoCapture(0)

controller = Controller(cap)


# Full-screen mode
cv2.namedWindow("Game", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Game",cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN)


controller.welcome(TIME_WELCOME)
state = controller.ask()

while state == "Continue":
    controller.countdown(TIME_PER_ROUND)
    controller.game_play(TIME_HOLD_SIGNAL)
    state = controller.ask()

cap.release()
cv2.destroyAllWindows()
