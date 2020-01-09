"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0) # ORGINAL CODE
#nwebcam = cv2.VideoCapture(-1) # ADAM CODE

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

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", frame)

    gaze.annotated_frame_eye(frame) 
    gaze.annotated_gaze_estimation_visual(frame)


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