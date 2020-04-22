import numpy as np
import cv2 as cv

# define a nothing function for tackbar 
def nothing(x):
    pass

# Create a black image, a window
img = np.zeros((300, 512, 3), np.uint8)

# name window
cv.namedWindow('image')

# create trackbars for color change
cv.createTrackbar('R', 'image', 0, 255, nothing)
cv.createTrackbar('G', 'image', 0, 255, nothing)
cv.createTrackbar('B', 'image', 0, 255, nothing)

# create switch title for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
# create trackbars for switch change
cv.createTrackbar(switch, 'image', 0, 1, nothing)

# create a loop
while(1):
    # show img as frame
    cv.imshow('image', img)

    # wait 1ms as frame update interval
    k = cv.waitKey(1) & 0xFF

    # listen pressed key 'esc' break loop
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv.getTrackbarPos('R', 'image')
    g = cv.getTrackbarPos('G', 'image')
    b = cv.getTrackbarPos('B', 'image')
    # get ON/OFF flag
    s = cv.getTrackbarPos(switch, 'image')

    if s == 0:
        # if OFF, show black
        img[:] = 0
    else:
        # if ON, show (b, g, r) color
        img[:] = [b, g, r]
# close 
cv.destroyAllWindows()
