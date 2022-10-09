import numpy as np
import random

np.random.seed(0)

class Computer:
    def __init__(self):
        # Initialy, computer's point = 0 and signal = 'None'
        self.score = 0
        self.signal = 'None'

    # This function randomly selects from Rock, Paper, and Scissors
    # to use as computer signal
    def makeSignal(self, labels):
        self.signal = random.choice(labels[1:4])

        
    