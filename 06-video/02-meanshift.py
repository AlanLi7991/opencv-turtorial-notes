import numpy as np
import cv2 as cv
import argparse
parser = argparse.ArgumentParser(description='''
This sample demonstrates the meanshift algorithm. The example file can be downloaded from:
https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4
''')
parser.add_argument('--image', type=str, help='path to image file', default='slow_traffic_small.mp4')
args = parser.parse_args()

# read video
cap = cv.VideoCapture(args.image)

# take first frame of the video
ret, frame = cap.read()

# setup initial location of window
x, y, w, h = 300, 200, 100, 50  # simply hardcoded the values
track_window = (x, y, w, h)

# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]

# convert to hsv 
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

# create mask with threshold hue 0~180, saturate 60~255, value 32~255
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

# use first channel hue create histogram
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])

# normalize histogram to int value
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)


while(True):
    # read frame 
    ret, frame = cap.read()
    if ret == True:
        # convert to hsv
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # create back project of frame
        bp = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(bp, track_window, term_crit)

        # get track window size
        x, y, w, h = track_window

        # draw rect angle on frame
        result = cv.rectangle(frame, (x, y), (x+w, y+h), 255, 2)

        # show result 
        cv.imshow('result', result)

        # wait 30ms
        k = cv.waitKey(30) & 0xff

        # use esc quit
        if k == 27:
            break
    else:
        break
