import numpy as np
import cv2 as cv

# TOPIC-1:
# Lucas-Kanade method computes optical flow for a sparse feature set 
# (in our example, corners detected using Shi-Tomasi algorithm).
#
# TOPIC-2:
# Dense optical flow computes the optical flow for all the points in the frame.
#

# read video
cap = cv.VideoCapture("vtest.avi")
# Take first frame and find corners in it
ret, prev_frame = cap.read()
# convert old frame to gray
prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

# create zero hsv space to store result
hsv = np.zeros_like(prev_frame)

# assign saturate to 255
hsv[..., 1] = 255

# loop
while(True):
    # continue read frame
    ret, frame = cap.read()

    # convert gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # create optical flow with Gunner Farneback's algorithm
    # return result is a 2-channel array with optical flow vectors (u,v)
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # calculates the magnitude and angle of (u, v)
    # 𝚖𝚊𝚐𝚗𝚒𝚝𝚞𝚍𝚎(I)=sqrt(𝚡(I)^2+𝚢(I)^2)
    # 𝚊𝚗𝚐𝚕𝚎(I)=𝚊𝚝𝚊𝚗𝟸(𝚢(I),𝚡(I))[⋅180/π]
    # u means dx/dt, which x changes during time 
    # v means dy/dt, which y changes during time
    #
    # which mean convert "cartesian coordinate system" to "polar system"
    # in order to represent it use hsv color space
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # convert to hue from angle
    hsv[..., 0] = ang*180/np.pi/2

    # convert to value from magnitude
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    # convert hsv to bgr
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    # show bgr 
    cv.imshow('frame', bgr)

    # wait 30ms
    k = cv.waitKey(30) & 0xff
    # use esc quit
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png', frame)
        cv.imwrite('opticalhsv.png', bgr)
    
    prev_gray = gray
