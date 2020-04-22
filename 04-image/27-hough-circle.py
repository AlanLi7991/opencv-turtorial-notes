import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread('logo.jpg', cv.IMREAD_COLOR)

# convert to gray
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# blur
blur = cv.medianBlur(gray, 5)

# get circles
circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT, 1, 20, param1=150, param2=30, minRadius=0, maxRadius=0)

# around to make it integer
circles_around = np.around(circles)

# convert float32 to uint16
circles_uint16 = np.uint16(circles_around)

# make result
result = img.copy()

# loop
for circle in circles_uint16[0, :]:
    # get center
    center = tuple(circle[0:2])
    # get radius
    radius = int(circle[2])
    # draw the outer circle
    cv.circle(result, center, radius, (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle(result, center, 2, (0, 0, 255), 3)

# show
plt.subplot(2, 2, 1), plt.imshow(img), plt.title("original")
plt.subplot(2, 2, 2), plt.imshow(blur, "gray"), plt.title("blur")
plt.subplot(2, 2, 3), plt.imshow(gray, "gray"), plt.title("gray")
plt.subplot(2, 2, 4), plt.imshow(result), plt.title("result")
plt.show()
