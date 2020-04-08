"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.mouse import Mouse
from datetime import datetime
import time

gaze = GazeTracking()
webcam = cv2.VideoCapture(0) # ORGINAL CODE
cursor = Mouse()
#nwebcam = cv2.VideoCapture(-1) # ADAM CODE
username = input('What is your name: ')
print('Get ready. Look at your cursor {} and move it around!'.format(username))
time.sleep(2)

#--------------------------


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

#    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    mouse_x, mouse_y = cursor.cursor_position()
#    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
#    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
#    cv2.putText(frame, "cursor x position: " + str(mouse_x), (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
#    cv2.putText(frame, "cursor y position: " + str(mouse_y), (90, 235), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

#---------------------------------------------------------
    
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
                x_value += 20
            else:
                x_value = 0
                y_value += 100
        else:
            do_calibration = False


#--------------------------------------------------------
#    gaze.annotated_frame_eye(frame) # displays landmark points
#    gaze.annotated_gaze_estimation_visual(frame) # displays spot looked at on screen
#    gaze.log_landmarks_pupils_and_cursor_pos(username)

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