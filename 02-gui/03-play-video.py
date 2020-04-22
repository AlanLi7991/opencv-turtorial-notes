import numpy as np
import cv2 as cv

# read video file by name
cap = cv.VideoCapture('vtest.avi')

# create a loop play frame
while cap.isOpened():
    # read frames of video
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # wait 1ms means update frame interval
    cv.imshow('frame', gray)
    # if pressed key is 'q', break the loop
    if cv.waitKey(1) == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
