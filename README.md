# Hand-Gesture-Recognition-System
=======
# Hand Gesture Recognition System using CNN :white_heart:
### :boy: Krishna Basak :boy: Kesavan R :boy: Thevaprakash P :boy: Shinjan Verma
---------
This repository contains all the `codes` for our project titled as **Hand Gesture Recognition System using CNN**. The project focuses on -
1. Making a Convulation Neural Network (CNN) and train the model with custom dataset (images of gestures).
2. Getting live gesture recgnition through webcam.
3. Do respective task according to recognized gesture. _(Optional)_

## Requirements
1. OpenCV
2. NumPy
3. Scikit-Learn
4. Glob
5. PyAutoGUI
6. Tensorflow
7. Keras
8. Matplotlib

## Work Explanation
[x] [PalmTracker.py](https://github.com/kr-basak/Hand-Gesture-Recognition-System/blob/main/PalmTracker.py) : Use this file to make your own image dataset for training and testing the CNN model. Make necessary changes in *number of images* and *path to image* and then run the proram.
[x] [resize.py](https://github.com/kr-basak/Hand-Gesture-Recognition-System/blob/main/resize.py) : Set appropriate path where you previously saved the images and run the code to resize/reshape them for training and testing.
[x] [num-trainer.ipynb](https://github.com/kr-basak/Hand-Gesture-Recognition-System/blob/main/num-trainer.ipynb) : Run the file to define the *Keras* model and train-test the model using the images.
[x] [num-recog.py](https://github.com/kr-basak/Hand-Gesture-Recognition-System/blob/main/num-recog.py) : This file will predict/recognize hand gesture realtime.
[ ] [controll.py](https://github.com/kr-basak/Hand-Gesture-Recognition-System/blob/main/controll.py) : This is an optional functionality. If you want to controll your PC through hand gestures, this might help you. To use this functionality - 1. Import this file in [num-recog.py](https://github.com/kr-basak/Hand-Gesture-Recognition-System/blob/main/num-recog.py) and add `controll(predicted_class)` to the line number **149**.