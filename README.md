> cd Rock-Paper-Scissors
# Training model

## Collect data

> python model/collect_data.py

Show your hand gesture and press the corresponding key (follow the on-screen instructions) to save the recording. End data recording by pressing the ESC key. The records will be saved at model/keypoint_data.csv

## Training

Run  **model/model_training.ipynb** for training model

## Evaluate

> python model/real_time_predict.py

Show your hand gesture and the system will display the result in the left corner of the screen

# Play game

> python game.py

And follow the instructions on the screen

