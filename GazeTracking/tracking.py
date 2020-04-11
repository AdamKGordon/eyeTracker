"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.mouse import Mouse
from pynput.mouse import Button,Controller
import time
mouse = Controller()
from tensorflow.keras import layers
import numpy as np
from time import sleep
#import threading
#from gaze_tracking.voice import voice
import pandas as pd


from tensorflow.keras.models import load_model

#voice_Thread = threading.Thread(target=voice, args=(), daemon=True)
#voice_Thread.start()

gaze = GazeTracking()
webcam = cv2.VideoCapture(0) # ORGINAL CODE
webcam.set(cv2.CAP_PROP_FPS, 10)
cursor = Mouse()
#webcam = cv2.VideoCapture(-1) # ADAM CODE
username = input('What is your name: ')
print('Get ready. Look at your cursor {} and move it around!'.format(username))
time.sleep(2)
model_x = load_model('predict_xindex.h5')
model_y = load_model('predict_yindex.h5')
#model = load_model('model3.h5')
#webcam = cv2.VideoCapture(0)
#width = 1920
#height = 1080
#webcam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


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
    
    ll = gaze.get_landmark_points_left()
    rl = gaze.get_landmark_points_right()
    

#    if left_pupil:
#        xnew = [[[left_pupil[0],right_pupil[0], #1535.2 863.2 960
#              ll[0].x,ll[1].x,ll[2].x,
#              ll[3].x,ll[4].x,ll[5].x,
#              rl[0].x,rl[1].x,rl[2].x,
#              rl[3].x,rl[4].x,rl[5].x]]]
    
#    if left_pupil:
#        ynew = [[[left_pupil[1],right_pupil[1],
#              ll[0].y,ll[1].y,ll[2].y,
#              ll[3].y,ll[4].y,ll[5].y,
#              rl[0].y,rl[1].y,rl[2].y,
#              rl[3].y,rl[4].y,rl[5].y]]]
    if left_pupil:
        xnew = [left_pupil[0],left_pupil[1],right_pupil[0],right_pupil[1],#1535.2 863.2 960
              ll[0].x,ll[0].y,ll[1].x,ll[1].y,ll[2].x,ll[2].y,
              ll[3].x,ll[3].y,ll[4].x,ll[4].y,ll[5].x,ll[5].y,
              rl[0].x,rl[0].y,rl[1].x,rl[1].y,rl[2].x,rl[2].y,
              rl[3].x,rl[3].y,rl[4].x,rl[4].y,rl[5].x,rl[5].y]
        
    train_dataset = pd.read_csv("test_input.csv")
    train_dataset.columns = ["lx", "ly", "rx", "ry", "ll1x", "ll1y", "ll2x", "ll2y", "ll3x", "ll3y", "ll4x", "ll4y", "ll5x", "ll5y"
    , "ll6x", "ll6y", "rl1x", "rl1y", "rl2x", "rl2y", "rl3x", "rl3y", "rl4x", "rl4y", "rl5x", "rl5y", "rl6x", "rl6y"]
    
    # Get the stats of input data
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()
        
    current_data = pd.DataFrame(xnew).transpose()
    current_data.columns = train_dataset.columns
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']
    
    normed_current_data = norm(current_data)
    
    xpred = model_x.predict(normed_current_data)
    ypred = model_y.predict(normed_current_data)
    
    print ("x:",xpred,",y:",ypred)
    green = (0,0,255)
    cv2.circle(frame, (int(xpred), int(ypred)), 2, green)

    xpred = ((xpred*1920.0)/1535.2)
    ypred = ((ypred*1080.0)/863.2)
#    xpred = ((ynew[0][0]*1920.0)/1535.2)
#    ypred = ((ynew[0][0]*1080.0)/863.2) 
    
#    xpred = ((xpred-850.0)*1920.0)/200.0    
#    ypred = ((ypred-520.0)*1080.0)/100.0
    
 #   print(abs(xpred[0][0]-float(mouse.position[0]))>300.0 or abs(ypred[0][0]-float(mouse.position[1]))>300.0)
    
    # if(abs(xpred[0][0]-float(mouse.position[0]))>300.0 or abs(ypred[0][0]-float(mouse.position[1]))>150.0):
    #     mouse.position=(1920-xpred[0][0],ypred[0][0])
 
#   if(abs(xpred-float(mouse.position[0]))>300.0 or abs(ypred-float(mouse.position[1]))>150.0):
#        mouse.position=(1920-xpred,ypred)

    #print(mouse.position)
    
#    mouse_x, mouse_y = cursor.cursor_position()
#    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
#    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
#    cv2.putText(frame, "cursor x position: " + str(mouse_x), (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
#    cv2.putText(frame, "cursor y position: " + str(mouse_y), (90, 235), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    gaze.annotated_frame_eye(frame) # displays landmark points
    # gaze.annotated_gaze_estimation_visual(frame) # displays spot looked at on screen
 #   gaze.log_landmarks_pupils_and_cursor_pos(username)
    
    
    
    

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow("window", frame)
    
    sleep(0.1)



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
