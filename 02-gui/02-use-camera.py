import numpy as np
import cv2 as cv

# ATTENTION(for Mac):
# if you use system higher than 10.14 
# you need authorize the terminal use camera at first
# System Preferences - Security & Privacy - Privacy - Camera
#
# from now on, must use 3rd promgram to get it 
# https://stackoverflow.com/questions/56084303/opencv-command-line-app-cant-access-camera-under-macos-mojave
#

# normally one camera will be connected (as in laptop). 
# so I simply pass 0 (or -1)
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('frame', gray)
    # wait 1ms means update frame interval
    if cv.waitKey(1) == ord('q'):
        # if pressed key is 'q', break the loop
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
