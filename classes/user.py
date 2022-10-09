import numpy as np


class User:
    def __init__(self):
        # Initialy, user's point = 0 and signal = 'None'
        self.score = 0
        self.signal = 'None'

    # This function takes the signal parameter 
    # (model predicted result) as a user signal
    def makeSignal(self, signal):
        self.signal = signal
    