import numpy as np
import cv2 as cv

# true if mouse is pressed
drawing = False 
# if True, draw rectangle. Press 'm' to toggle to curve
mode = True  
# pixels
ix, iy = -1, -1

# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode
    # record pixels when button down
    if event == cv.EVENT_LBUTTONDOWN:
        # set drawing flag
        drawing = True
        ix, iy = x, y
    # paint when mouse move
    elif event == cv.EVENT_MOUSEMOVE:
        # if set flag true 
        if drawing == True:
            # draw rectangle or circle by mode
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)
    # cancel drawing flag after button up
    elif event == cv.EVENT_LBUTTONUP:
        # cancel drawing
        drawing = False
        # draw last pixel
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)

# create a black image
img = np.zeros((512, 512, 3), np.uint8)
# give window name
cv.namedWindow('image')
# bind the event function to window
cv.setMouseCallback('image', draw_circle)

# create a loop update with interval
while(True):
    # show img as frame
    cv.imshow('image', img)
    # use 1ms as a frame
    k = cv.waitKey(1) & 0xFF
    # check pressed key is "m"
    if k == ord('m'):
        # switch rectangle/circle mode 
        mode = not mode
    # check pressed key is "esc"
    elif k == 27:
        # break loop
        break
    
# close
cv.destroyAllWindows()
