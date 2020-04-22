import numpy as np
import cv2 as cv
import argparse

# TOPIC-1:
# Lucas-Kanade method computes optical flow for a sparse feature set
# (in our example, corners detected using Shi-Tomasi algorithm).
#
# TOPIC-2:
# Dense optical flow computes the optical flow for all the points in the frame.
#

parser = argparse.ArgumentParser(description='''
This sample demonstrates Lucas-Kanade Optical Flow calculation. The example file can be downloaded from:
https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4
''')
parser.add_argument('--image', type=str, help='path to image file', default='slow_traffic_small.mp4')
args = parser.parse_args()

# read video
cap = cv.VideoCapture(args.image)
# Take first frame and find corners in it
ret, prev_frame = cap.read()
# convert old frame to gray
prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

# find keypoints
prev_pts = cv.goodFeaturesToTrack(
    prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
)

# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Create a mask image for drawing purposes
mask = np.zeros_like(prev_frame)

# create criteria to stop
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)

while(True):
    # continue read frame
    ret, frame = cap.read()

    # convert gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    # pts :
    #   output vector of 2D points (with single-precision floating-point coordinates) 
    #   containing the calculated new positions
    # status :
    #   output status vector (of unsigned chars); 
    #   each element of the vector is set to 1 
    #   if the flow for the corresponding features has been found, 
    #   otherwise, it is set to 0.
    # err :
    #   output vector of errors; 
    #   each element of the vector is set to an error for the corresponding feature, 
    #   type of the error measure can be set in flags parameter; 
    #   if the flow wasn't found then the error is not defined
    # 
    pts, status, err = cv.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts, None, winSize=(15, 15), maxLevel=2, criteria=criteria
    )

    # Select good points
    metchs = pts[status == 1]
    metchs_prev = prev_pts[status == 1]

    # draw the tracks
    for i, (cur, prev) in enumerate(zip(metchs, metchs_prev)):
        # unpack point from ravel
        x1, y1 = cur.ravel()
        x0, y0 = prev.ravel()
        # draw circle on current frame
        frame = cv.circle(frame, (x1, y1), 5, color[i].tolist(), -1)
        # draw line on mask, record the tracks
        mask = cv.line(mask, (x0, y0), (x1, y1), color[i].tolist(), 2)
    # apply mask on current frame
    img = cv.add(frame, mask)

    # show processed frame
    cv.imshow('frame', img)

    # wait 30ms
    k = cv.waitKey(30) & 0xff

    # use esc quit
    if k == 27:
        break

    # Now update the previous frame and previous points
    prev_gray = gray.copy()
    prev_pts = metchs.reshape(-1, 1, 2)
