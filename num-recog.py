import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam
import cv2
import numpy as np
import os
import datetime
from skimage import io
import os
import random
import matplotlib.pyplot as plt
import glob

bg= None

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def _load_weights():
    try:
        model = keras.models.load_model("numbers.h5")
        print(model.summary())
        return model
    except Exception as e:
    	print(e)
    	return None

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

def getPredictedClass(model):
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (100, 120))
    gray_image = gray_image.reshape(1, 100, 120, 1)
    prediction = model.predict(np.asarray(gray_image))
    prd=''
    predicted_class = np.argmax(prediction)
    if predicted_class == 0:
        prd= "ONE"
    elif predicted_class == 1:
        prd= "TWO"
    elif predicted_class == 2:
        prd= "THREE"
    elif predicted_class == 3:
        prd= "FOUR"
    elif predicted_class == 4:
        prd= "FIVE"
    elif predicted_class == 5:
        prd= "BLANK"
    print(prd)
    return prd

record = False

if __name__ == "__main__":
    accumWeight = 0.5
    camera = cv2.VideoCapture(0)
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    calibrated = False
    model = _load_weights()
    k = 0
    while (True):
        (grabbed, frame) = camera.read()
        frame = cv2.resize(frame, (700,700))
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("\n\n\n[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
                print('------------------------------------------------------------------------------------')

        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if (k % (fps / 6) == 0) and record == True:
                    cv2.imwrite('temp.png', thresholded)
                    predictedClass = getPredictedClass(model)
                    showData = False
                    cv2.putText(clone, str(predictedClass), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Thesholded", thresholded)
        k = k + 1
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then terminate
        if keypress == ord("q"):
            break
        # if the user pressed "s", then start/resume predicting
        if keypress == ord("s"):
        	record = True
        # if the user pressed "p", then stop predicting
        if keypress == ord("p"):
        	record = False

    camera.release()
    cv2.destroyAllWindows()