import numpy as np
import cv2 as cv

# mouse callback function
# MacOS track pad use LeftButtonDown Event
# Mouse on windows use LeftButtonDoubleClick Event
def draw_circle(event, x, y, flags, param):
    mouse_down = event == cv.EVENT_LBUTTONDOWN
    double_click = event == cv.EVENT_LBUTTONDBLCLK
    if mouse_down or double_click:
        cv.circle(img, (x, y), 100, (255, 0, 0), -1)


# Create a black image, a window and bind the function to window
img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')

# register mouse click event
cv.setMouseCallback('image', draw_circle)

# create a loop update with interval
while(True):
    # show img as frame
    cv.imshow('image', img)
    # use 20ms as a frame
    if cv.waitKey(20) & 0xFF == 27:
        break

# close
cv.destroyAllWindows()
