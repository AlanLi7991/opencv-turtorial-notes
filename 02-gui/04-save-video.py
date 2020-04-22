import numpy as np
import cv2 as cv

# capture video from default camera
cap = cv.VideoCapture(0)

# Define the codec
# A FourCC ("four-character code") is a sequence of four bytes (typically ASCII) used to uniquely identify data formats
# all possible list:
# http://www.fourcc.org/codecs.php
fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')

# create size of capture
# VideoWriter size must equal to the camera resolution
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (width, height)

# create VideoWriter object
out = cv.VideoWriter('output.mp4', fourcc, 20.0, size)

# create a loop
while cap.isOpened():
    # read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # flip the frame
    # because the camera capture is mirror of real position
    # 0 means flipping around the x-axis
    # 1 means flipping around y-axis.
    # -1 means flipping around both axes.
    frame = cv.flip(frame, 1)

    # write the flipped frame
    out.write(frame)

    # dispay frame
    cv.imshow('frame', frame)

    # update frame with 1ms interval
    # listen press key 'q' to break
    if cv.waitKey(1) == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv.destroyAllWindows()
