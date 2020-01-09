from __future__ import division
import os
import cv2
import dlib

#import eye
#from eye import Eye
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None 
        self.eye_left = None
        self.eye_right = None
        self.landmarks = None
        self.algorithm_4_points = None
        self.calibration = Calibration()

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(
        #    "C:\\Users\\adamk\\Desktop\\University\\Y4\\Capstone-\\GazeTracking\\gaze_tracking",
            cwd,
            "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left  = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)
            self.landmarks = landmarks # face landmarks for facial tracking

        except IndexError:
            self.eye_left  = None
            self.eye_right = None
            self.landmarks = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()


    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def mid_point(self, p1, p2):
        """Returns the middle point (x,y) between two points

        Arguments:
            p1 (dlib.point): First point
            p2 (dlib.point): Second point
        """
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return (x, y)

    def eye_box_right_coords(self):
        """Returns the 4 (left,right,top,bot) coordinates of the left eye"""
        if self.pupils_located:
            left = (self.landmarks.part(self.eye_right.points[0]).x, self.landmarks.part(self.eye_right.points[0]).y)
            right = (self.landmarks.part(self.eye_right.points[3]).x, self.landmarks.part(self.eye_right.points[3]).y)
            top = self.mid_point(self.landmarks.part(self.eye_right.points[1]), self.landmarks.part(self.eye_right.points[2]))
            bottom = self.mid_point(self.landmarks.part(self.eye_right.points[5]), self.landmarks.part(self.eye_right.points[4]))
            return (left, right, top, bottom)

    def get_landmark_points_left(self): # left
        if self.pupils_located:
            point_list = []
            for i in range(len(self.eye_left.points)):
                point_list.append(self.landmarks.part(self.eye_left.points[i]))
            
            return tuple(point_list)

    def get_landmark_points_right(self): # right
        if self.pupils_located:
            point_list = []
            for i in range(len(self.eye_right.points)):
                point_list.append(self.landmarks.part(self.eye_right.points[i]))
            
            return tuple(point_list)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame_eye(self, frame):
        green = (0,0,255)
        if self.pupils_located:
            landmark_points_left  = self.get_landmark_points_left()
            landmark_points_right = self.get_landmark_points_right()
            if (1): #left eye outline points
                for point in landmark_points_left:
                    cv2.circle(frame, (int(point.x), int(point.y)), 1, green)

            if (1): #right eye rectangle points
                for point in landmark_points_right:
                    cv2.circle(frame, (int(point.x), int(point.y)), 1, green)
                #right_rectange_points = self.eye_box_right_coords()
                #for point in right_rectange_points:
                #    cv2.circle(frame, point, 1, green)

    def annotated_gaze_estimation_visual(self, frame):
        # annotates where we think on the screen you are looking
        frame_height, frame_width = frame.shape[:2]
        blue = (255,0,0)
        purple = (200,0,255)
        if self.pupils_located:
            if(0): # algorithm 1 (on left eye only, up and down)
                   # top to bot mapping 10%none 55%mapped 35%none. this 65% of the pupil zone is mapped to the screen
                   #  

                landmark_points = self.get_landmark_points_left()
                pupil_point     = self.pupil_left_coords()

                top = (int(landmark_points[1].y) + int(landmark_points[2].y)) / 2
                bot = (int(landmark_points[4].y) + int(landmark_points[5].y)) / 2
                
                #print(str(top) + '  ' + str(pupil_point[1]))
                pupil_dist_from_top = pupil_point[1] - top #bottom of screen is 1
                pupil_percent_from_top = (pupil_dist_from_top)/(bot-top)
                #print(str(pupil_percent_from_top) + '   ' + str(bot-top))
                if pupil_percent_from_top < 0.1:
                    mapped_percent_y = 0
                elif pupil_percent_from_top > 0.65:
                    mapped_percent_y = 1
                else:
                    mapped_percent_y = (pupil_percent_from_top-0.1)/0.55

                cv2.circle(frame, (int(frame_width*0.5),int(frame_height*mapped_percent_y)), 1, blue)
                cv2.circle(frame, (int(frame_width*0.5),int(frame_height*mapped_percent_y)), 4, blue)

            if(0): #algorithm 2 using their ratios  
                point = (int(self.horizontal_ratio()*frame_width), int(self.vertical_ratio()*frame_height))
                cv2.circle(frame, point, 1, purple)
                cv2.circle(frame, point, 5, purple)


            if(0): # algorithm 3 (on left eye only, up and down)
                   # uses ratio of height and width 
                   #  

                landmark_points = self.get_landmark_points()
                pupil_point     = self.pupil_left_coords()

                top_vert = (int(landmark_points[1].y) + int(landmark_points[2].y)) / 2.0
                bot_vert = (int(landmark_points[4].y) + int(landmark_points[5].y)) / 2.0
                height_vert = bot_vert - top_vert
                left_vert  = int(landmark_points[0].x)
                right_vert = int(landmark_points[3].x)
                width_vert = right_vert - left_vert

                ratio_vert = 1 - height_vert/width_vert #percent from bottom
                #print(ratio_vert)
                shift_vert = -0.02
                calbotvert = 0.83 + shift_vert # to lower the est values, lower the cal vals
                calmidvert = 0.7  + shift_vert
                caltopvert = 0.67 + shift_vert
                ratio_scaled_vert = (ratio_vert-caltopvert)/(calbotvert-caltopvert)
                #print(ratio_scaled_vert)
                if ratio_scaled_vert > 1:
                    ratio_scaled_vert = 1
                gaze_vert_y = ratio_scaled_vert*frame_height

                left_hor  = (int(landmark_points[1].x) + int(landmark_points[5].x)) / 2.0
                right_hor = (int(landmark_points[2].x) + int(landmark_points[4].x)) / 2.0
                width_hor = right_hor - left_hor

                # eyes are closer to middle of face, this is left eye so move pupil point further from middle
                # Adding not subtracting because left and right is mirrored... TODO test this
                pupil_point_width_shifted = pupil_point[0] + 0.00*frame_width
                #print(pupil_point_width_shifted)
                percent_from_the_left_of_square = (pupil_point_width_shifted-left_hor)/(right_hor-left_hor)
                gaze_hor_x =  (1.0 - percent_from_the_left_of_square) * frame_width
                #print(gaze_hor_x)
                
                #cv2.circle(frame, (int(frame_width*0.5), int(gaze_vert_y)), 1, (0,255,0))
                #cv2.circle(frame, (int(frame_width*0.5), int(gaze_vert_y)), 4, (0,255,0))

                #cv2.circle(frame, (int(gaze_hor_x), int(frame_height*0.5)), 1, (0,255,0))
                #cv2.circle(frame, (int(gaze_hor_x), int(frame_height*0.5)), 4, (0,255,0))

                cv2.circle(frame, (int(gaze_hor_x), int(gaze_vert_y)), 1, (0,255,0))
                cv2.circle(frame, (int(gaze_hor_x), int(gaze_vert_y)), 4, (0,255,0))
                cv2.circle(frame, (int(gaze_hor_x), int(gaze_vert_y)), 7, (0,255,0))

            if(0): # algorithm 4, algorithm 3 but weighted sample yi(t)=Aix(t)+Biyi(t-1), yi(t)=self.algorithm_4_points[i]
                if self.algorithm_4_points == None:
                    self.algorithm_4_points = [[frame_height/2, frame_width/2]]*29 # *num of points
                landmark_points = self.get_landmark_points()
                pupil_point     = self.pupil_left_coords()

                top_vert = (int(landmark_points[1].y) + int(landmark_points[2].y)) / 2.0
                bot_vert = (int(landmark_points[4].y) + int(landmark_points[5].y)) / 2.0
                height_vert = bot_vert - top_vert
                left_vert  = int(landmark_points[0].x)
                right_vert = int(landmark_points[3].x)
                width_vert = right_vert - left_vert

                ratio_vert = 1 - height_vert/width_vert #percent from bottom
                #print(ratio_vert)
                shift_vert = -0.02
                calbotvert = 0.83 + shift_vert # to lower the est values, lower the cal vals
                calmidvert = 0.7  + shift_vert
                caltopvert = 0.67 + shift_vert
                ratio_scaled_vert = (ratio_vert-caltopvert)/(calbotvert-caltopvert)
                #print(ratio_scaled_vert)
                if ratio_scaled_vert > 1:
                    ratio_scaled_vert = 1
                gaze_vert_y = ratio_scaled_vert*frame_height

                left_hor  = (int(landmark_points[1].x) + int(landmark_points[5].x)) / 2.0
                right_hor = (int(landmark_points[2].x) + int(landmark_points[4].x)) / 2.0
                width_hor = (right_hor - left_hor)

                # eyes are closer to middle of face, this is left eye so move pupil point further from middle
                # Adding not subtracting because left and right is mirrored... TODO test this
                pupil_point_width_shifted = pupil_point[0] + 0*0.005*frame_width #use a multiple less than 0.01
                #print(pupil_point_width_shifted)
                percent_from_the_left_of_square = (pupil_point_width_shifted-left_hor)/width_hor
                gaze_hor_x =  (1.0 - percent_from_the_left_of_square) * frame_width
                #print(gaze_hor_x)
                
                algo_4_num_pts = len(self.algorithm_4_points)
                color_step = 255/algo_4_num_pts
                a = 0.07
                for k in range(algo_4_num_pts):
                    csk = color_step * k #colour step * k
                #                               yi(t)=Ai*x(t)+Bi*yi(t-1) ,ai=a*i ...
                    self.algorithm_4_points[k] = [a*k*gaze_hor_x  + (1-a*k)*self.algorithm_4_points[k][0],
                                                  a*k*gaze_vert_y + (1-a*k)*self.algorithm_4_points[k][1]]
                    tmp_pt_x = int(self.algorithm_4_points[k][0])
                    tmp_pt_y = int(self.algorithm_4_points[k][1])
                    if k < algo_4_num_pts*1:
                        cv2.circle(frame, (tmp_pt_x, tmp_pt_y), 1, (csk,csk,csk))
                        cv2.circle(frame, (tmp_pt_x, tmp_pt_y), 4, (csk,csk,csk))
                        cv2.circle(frame, (tmp_pt_x, tmp_pt_y), 7, (csk,csk,csk))  
              

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
            self.annotated_frame_eye(frame) 
            self.annotated_gaze_estimation_visual(frame)
        return frame
