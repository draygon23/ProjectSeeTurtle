import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([90,110,90])
    upper_blue = np.array([150,255,255])

    # Threshold the HSV image to get only blue colors
    blueMask = cv2.inRange (hsv, lower_blue, upper_blue)
    bluecnts = cv2.findContours(blueMask.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(bluecnts)>0:
        blue_area = max(bluecnts, key=cv2.contourArea)
        (blue_xg,blue_yg,blue_wg,blue_hg) = cv2.boundingRect(blue_area)
        cv2.rectangle(frame,(blue_xg,blue_yg),(blue_xg+blue_wg, blue_yg+blue_hg),(255,0,0),2)
        blueCenterCoord = (blue_xg + (blue_wg / 2), blue_yg + (blue_hg / 2))
        print("Blue center: ", blueCenterCoord)

    # define range of red color in HSV
    lower_red = np.array([136, 87, 111])
    upper_red = np.array([180, 255, 255])

    # Threshold the HSV image to get only red colors
    redMask = cv2.inRange(hsv, lower_red, upper_red)
    redcnts = cv2.findContours(redMask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(redcnts) > 0:
        red_area = max(redcnts, key=cv2.contourArea)
        (red_xg, red_yg, red_wg, red_hg) = cv2.boundingRect(red_area)
        cv2.rectangle(frame, (red_xg, red_yg), (red_xg + red_wg, red_yg + red_hg), (0, 0, 255), 2)
        redCenterCoord = (red_xg + (red_wg / 2), red_yg + (red_hg / 2))
        print("Red center: ", redCenterCoord)

    # define range of yellow color in HSV
    lower_yellow = np.array([22, 60, 200])
    upper_yellow = np.array([60, 255, 255])

    # Threshold the HSV image to get only yellow colors
    yellowMask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellowcnts = cv2.findContours(yellowMask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(yellowcnts) > 0:
        yellow_area = max(yellowcnts, key=cv2.contourArea)
        (yellow_xg, yellow_yg, yellow_wg, yellow_hg) = cv2.boundingRect(yellow_area)
        cv2.rectangle(frame, (yellow_xg, yellow_yg), (yellow_xg + yellow_wg, yellow_yg + yellow_hg), (0, 255, 255), 2)
        yellowCenterCoord = (yellow_xg + (yellow_wg / 2), yellow_yg + (yellow_hg / 2))
        print("Yellow center: ", yellowCenterCoord)
    cv2.imshow('Frame',frame)
    cv2.imshow('Red mask',redMask)
    cv2.imshow('Blue mask',blueMask)
    cv2.imshow('Yellow mask',yellowMask)

    k = cv2.waitKey(5)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
