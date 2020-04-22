from __future__ import print_function
import cv2 as cv
import argparse

# prepare arguments and set default
parser = argparse.ArgumentParser(description='''
This program shows how to use background subtraction methods provided by OpenCV. You can process both videos and images.
''')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

# create background subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

# create capture of video frames 
capture = cv.VideoCapture(args.input)

# check capture is valid
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)

# while true 
while True:
    # read capture frame of 
    ret, frame = capture.read()
    
    # check frame valid
    if frame is None:
        break
    
    # use background subtract on frame get foreground mask
    fgMask = backSub.apply(frame)

    # get frame position
    position = capture.get(cv.CAP_PROP_POS_FRAMES)

    # create a rectangle on frame as label background
    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)

    # render position with string on frame as label text
    cv.putText(frame, str(position), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # show different frame on different window
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    # each frame wait 30ms 
    keyboard = cv.waitKey(30)

    # use q or ESC exit the program
    if keyboard == ord('q') or keyboard == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
