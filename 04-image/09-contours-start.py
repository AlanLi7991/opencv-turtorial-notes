import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread('box.jpg', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# threshold
ret, thresh = cv.threshold(gray, 127, 255, 0)

# find contours
# https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html
# document has error
# findContours had not changed source img and plus one return value at first
some, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# draw all contours
green = img.copy()
cv.drawContours(green, contours, -1, (0, 255, 0), 3)

# draw first contours
red = img.copy()
cv.drawContours(red, contours, 0, (255, 0, 0), 3)

# show
plt.subplot(2, 2, 1), plt.imshow(img, "gray")
plt.subplot(2, 2, 2), plt.imshow(green, "gray")
plt.subplot(2, 2, 3), plt.imshow(red, "gray")
plt.show()
