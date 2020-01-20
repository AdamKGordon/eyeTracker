"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.calibration import Calibration
from gaze_tracking.mouse import Mouse
from datetime import datetime
import time

gaze = GazeTracking()
webcam = cv2.VideoCapture(0) # ORGINAL CODE
cursor = Mouse()
#nwebcam = cv2.VideoCapture(-1) # ADAM CODE

#username = input('What is your name: ')
#username = "jacky"
#print('Get ready. Look at your cursor {} and move it around!'.format(username))
time.sleep(2)
#--------------------------


time = datetime.now()
do_calibration = True
max_x = 500
max_y = 500
x_value = 0
y_value = 0
#print time
#---------------------------


while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)


#---------------------------------------------------------
    if do_calibration:
 #       Calibration().calibrate_from_data("/Users/jeromicho/git/eyeTracker/input_data_Jacky.csv")
        if (datetime.now() - time).total_seconds() < 2:
            cv2.putText(frame, "+", (y_value, x_value), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
        else:
            print ("change")
            time = datetime.now()
            print (x_value)
            print (y_value)
            print (time)
            if (max_x - x_value <= 10 and max_y - y_value <= 10):
                print ("cal")
 #               calibration().calibrate_from_data(a)
                do_calibration = False
            elif (max_y - y_value <= 10):
                print ("x +")
                x_value = x_value + max_x/3
                y_value = 0
            else:
                print ("y +")
                y_value = (y_value + max_y/3)

#--------------------------------------------------------

  
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    mouse_x, mouse_y = cursor.cursor_position()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "cursor x position: " + str(mouse_x), (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "cursor y position: " + str(mouse_y), (90, 235), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    gaze.annotated_frame_eye(frame) # displays landmark points
    gaze.annotated_gaze_estimation_visual(frame) # displays spot looked at on screen
 #   gaze.log_landmarks_pupils_and_cursor_pos(username)

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", frame)



    # TODO
       # MOUSE CLASS: 
        # Methods:
            # Able to Capture current (x,y) of the cursor on the screen
            # Able to Move cursor to (x,y) on screen
            # Able to get dimensions of screen (or have this call another class called display to get display data like screen dimensions)
        # Idea for calibration (training model)
        #              enter calibration -> mouse auto moves around while user follows it with their eye ->
        #                           -> data is recorded (xy of mouse and pupil and landmarks points for users eye)
        #                               ->  outputs the eye points into input.csv and mouse xy into output.csv 
        #                                   -> feed into AI and AI outputs mouse points
              
    
    
    if cv2.waitKey(1) == 27:
        break
