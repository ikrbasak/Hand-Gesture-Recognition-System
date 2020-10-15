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
import pyautogui as p

# define the function
def controll(data):
	x, y = p.position()
	# print(x,y)  #for debugging
	n=p.onScreen(x,y)
	# print(n)  #for debugging
	if (n == True):
		if (data == 0):
			print('No task')
		elif (data == 1):
			p.press('down')
		elif (data == 2):
			p.press('up')
		elif (data == 3):
			p.press('volumedown')
		elif (data == 4):
			p.press('volumeup')
		elif (data == 5):
			print('nummguvytuyuy')
		else:
			print('No task assigned')


'''
import getdata as gt

md = keras.models.load_model("numbers.h5")

data_list=['one','two','three','four','five','blank']
i=0
j=0

for i in range(0, len(data_list)):
	path, label = gt.read(j)
	img=cv2.imread(path)
	img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img=cv2.resize(img, (100, 120))
	img=img.reshape(1, 100, 120, 1)
	pre=md.predict(np.asarray(img))
	pre=np.argmax(pre)
	print('----------- For *'+label+'* prediction is  :: ',pre)
	j+=1
	i+=1
'''

bg= None

#print('\n-----------------------------------------------------------\n')

# To segment the region of hand in the image
def segment(image, threshold=25):
    global bg
    # print(type(image))
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#print('\n-----------------------------------------------------------\n')

# load Model Weights
def _load_weights():
    try:
        model = keras.models.load_model("numbers.h5")
        # print(model.summary())
        # print(model.get_weights())
        # print(model.optimizer)
        return model
    except Exception as e:
    	print(e)
    	return None

#print('\n-----------------------------------------------------------\n')

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return
        
    cv2.accumulateWeighted(image, bg, aWeight)
#print('\n-----------------------------------------------------------\n')

def getPredictedClass(model):

    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (100, 120))

    gray_image = gray_image.reshape(1, 100, 120, 1)

    # cv2.imshow("test",gray_image)

    prediction = model.predict(np.asarray(gray_image))

    # prediction = model.predict(np.asarray(gray_image))

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
    # controll(predicted_class)
    return prd

#print('\n-----------------------------------------------------------\n')

record = False

if __name__ == "__main__":
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    #print('\n-----------------------------------------------------------\n')

    fps = int(camera.get(cv2.CAP_PROP_FPS))
    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590
    # initialize num of frames
    num_frames = 0
    # calibration indicator
    calibrated = False

    #print('\n-----------------------------------------------------------\n')

    model = _load_weights()
    k = 0
    # keep looping, until interrupted
    while (True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = cv2.resize(frame, (700,700))
        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        #print('\n-----------------------------------------------------------\n')

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our weighted average model gets calibrated
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("\n\n\n[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
                print('------------------------------------------------------------------------------------')

        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                #print('\n-----------------------------------------------------------\n')

                # count the number of fingers
                # fingers = count(thresholded, segmented)
                if (k % (fps / 6) == 0) and record == True:
                    cv2.imwrite('Temp.png', thresholded)
                    predictedClass = getPredictedClass(model)
                    showData = False
                    cv2.putText(clone, str(predictedClass), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # show the thresholded image
                cv2.imshow("Thesholded", thresholded)

                #print('\n-----------------------------------------------------------\n')

        k = k + 1
        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        #print('\n-----------------------------------------------------------\n')

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        if keypress == ord("s"):
        	record = True
        if keypress == ord("p"):
        	record = False

    # free up memory
    camera.release()
    cv2.destroyAllWindows()