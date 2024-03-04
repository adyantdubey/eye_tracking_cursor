import cv2
import numpy as np
import dlib
from math import hypot
import pyautogui

pyautogui.FAILSAFE = False
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y+p2.y)/2)


def get_blinking_ratio(eyepoints, faciallandmarks):
    left_point = (faciallandmarks.part(eyepoints[0]).x, faciallandmarks.part(eyepoints[0]).y)
    right_point = (faciallandmarks.part(eyepoints[3]).x, faciallandmarks.part(eyepoints[3]).y)
    center_top = midpoint(faciallandmarks.part(eyepoints[1]), faciallandmarks.part(eyepoints[2]))
    center_bottom = midpoint(faciallandmarks.part(eyepoints[5]), faciallandmarks.part(eyepoints[4]))

    # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio


def get_gaze_ratio(eyepoints, facial_landmarks):
    left_eye_region = np.array([
        (facial_landmarks.part(eyepoints[0]).x, facial_landmarks.part(eyepoints[0]).y),
        (facial_landmarks.part(eyepoints[1]).x, facial_landmarks.part(eyepoints[1]).y),
        (facial_landmarks.part(eyepoints[2]).x, facial_landmarks.part(eyepoints[2]).y),
        (facial_landmarks.part(eyepoints[3]).x, facial_landmarks.part(eyepoints[3]).y),
        (facial_landmarks.part(eyepoints[4]).x, facial_landmarks.part(eyepoints[4]).y),
        (facial_landmarks.part(eyepoints[5]).x, facial_landmarks.part(eyepoints[5]).y), ], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)

    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0:height, 0:int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0:height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    gaze_ratio = left_side_white / right_side_white
    return gaze_ratio

def move_cursor(gaze_ratio):
    screen_width, screen_height = pyautogui.size()
    cursor_region_width = screen_width*0.8
    cursor_region_height = screen_height*0.8

    new_x = int(gaze_ratio*cursor_region_width)
    new_y = int(cursor_region_height/2)

    pyautogui.moveTo(new_x,new_y)



font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        # x,y = face.left(), face.top()
        # x1,y1 = face.right(),  face.bottom()

        # cv2.rectangle(frame,(x,y), (x1,y1), (0,255,0) ,2)
        landmarks = predictor(gray, face)

        #blinking detection


        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47],landmarks)
        blinking_ratio = (left_eye_ratio+right_eye_ratio)/2


        if(blinking_ratio >5.5):
            cv2.putText(frame, "Blinking", (50,150), font, 3 , (255,0,0))


        #gaze Detection
        gaze_ratio_left_eye = get_gaze_ratio([36,37,38,39,40,41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42,43,44,45,46,47],landmarks)

        gaze_ratio = (gaze_ratio_left_eye+gaze_ratio_right_eye)/2

        move_cursor(gaze_ratio)

        cv2.putText(frame, str(gaze_ratio), (50,100), font, 2 ,(0, 0,255), 3)


    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)
    if  key == 27:
        break
    # if KeyboardInterrupt:
    #     break
cap.release()
cv2.destroyAllWindows()