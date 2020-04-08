"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""
import csv
import numpy as np
import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.mouse import Mouse
import time
from datetime import datetime

gaze = GazeTracking()
webcam = cv2.VideoCapture(0) # ORGINAL CODE

cursor = Mouse()
username = input('What is your name: ')
print('Get ready. Look at your cursor {} and move it around!'.format(username))
time.sleep(2)

InputData = []
OutputData = []

startingtime = datetime.now()
time = startingtime
do_calibration = True
max_x = 1300
max_y = 869
starting_x_value = 0
starting_y_value = 30
x_value = starting_x_value
y_value = starting_y_value
sec = 5

def saveData(filename, data):
    with open(filename, mode='w') as file_writer:
        file_writer = csv.writer(file_writer, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in data:
            file_writer.writerow(line)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    if (datetime.now() - startingtime).total_seconds() < 6:
        if (datetime.now() - time).total_seconds() >= 1:
            sec = sec - 1
            time = datetime.now()
        cv2.putText(frame, "Please look at the cursor", (400, 80), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        cv2.putText(frame, "Calibration starts in", (430, 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)    
        cv2.putText(frame, str(sec), (650, 250), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    elif do_calibration:
 #       Calibration().calibrate_from_data("/Users/jeromicho/git/eyeTracker/input_data_Jacky.csv")
        if y_value < max_y:
            if x_value < max_x:
                cv2.putText(frame, "+", (int(x_value), int(y_value)), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
                if gaze.pupils_located:
                    newInput = []
                    newOutput = []
                    newOutput.append(x_value)
                    newOutput.append(y_value)
                    landmark_points_left = gaze.get_landmark_points_left()
                    for point in landmark_points_left:
                        newInput.append(point.x)
                        newInput.append(point.y)
                        print ("left landmark: ", str(point.x), " ", str(point.y))
                    landmark_points_right = gaze.get_landmark_points_right()
                    for point in landmark_points_right:
                        newInput.append(point.x)
                        newInput.append(point.y)
                        print ("right landmark: ", str(point.x), " ", str(point.y))
                    x_left, y_left = gaze.pupil_left_coords()
                    newInput.append(x_left)
                    newInput.append(y_left)
                    x_right, y_right = gaze.pupil_right_coords()
                    newInput.append(x_right)
                    newInput.append(y_right)
                    print ("left pupil: ", str(x_left), " ", str(y_left))
                    print ("right pupil: ", str(x_right), " ", str(y_right))
                    print ("-----------------------------------")
                    InputData.append(newInput)
                    OutputData.append(newOutput)
                    print(len(InputData))
                x_value += 20
            else:
                x_value = 0
                y_value += 100
        else:
            cv2.putText(frame, "Calibration Compelete", (430, 200), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
            saveData('test_input.csv', InputData)
            saveData('test_output.csv', OutputData)
            do_calibration = False

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", frame)    

    if cv2.waitKey(1) == 27:
        saveData('test_input.csv', InputData)
        saveData('test_output.csv', OutputData)
        break

    # if cv2.waitKey(1) == 27:
    #     saveData('test_input.csv', InputData)
    #     saveData('test_output.csv', OutputData)
    #     break
