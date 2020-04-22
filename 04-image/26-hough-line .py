import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#
# Q1: What is "Hough Transform"?
# The Hough Transform is a popular technique to detect the various shape
#
# Q2: What is the keypoint of "Hough Transform"?
# if you can represent that shape in a mathematical form.
# It can detect the shape even if it is a little bit broken or distorted.
#
# Q3: What could the Hough algorithm do?
# 1. define a shape with mathematics
# 2. create a histogram container for statistics
# 3. loop every potential combination
# 4. take a threshold to get the result
#
# Q4: Why does "Hough Lines" use 1000 to draw?
# the mathematical define
# 1. use a polar coordinate system for statistics
# 2. target line is too normal for polar magnetic
# can not represent the length of a line
# so 1000 large is enough to cross the image
#
#

# read
img = cv.imread("gradients.jpg", cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# convert gray
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# find edges with canny
edges = cv.Canny(gray, 50, 150, apertureSize=3)

# find lines with hough transform
# the first param is ρ(rho), the maximum possible distance is the diagonal length of the image.
# So taking one pixel accuracy, the number of rows can be the diagonal length of the image.
# the second param is theta, from 0 ~ 180
# last parameter is threshold, if it is too bigger, it will return to None
img_result = img.copy()
lines = cv.HoughLines(edges, 1, np.pi/180, 120)

# loop line in array
for line in lines:
    rho, theta = line[0]
    # get sin/cos value
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    # get delta value
    delta_x = cos_theta*rho
    delta_y = sin_theta*rho
    # get point1,
    x1 = int(delta_x + 1000*(-sin_theta))
    y1 = int(delta_y + 1000*(cos_theta))
    # get point2,
    x2 = int(delta_x - 1000*(-sin_theta))
    y2 = int(delta_y - 1000*(cos_theta))
    cv.line(img_result, (x1, y1), (x2, y2), (255, 0, 0), 1)

# find lines with probabilistic hough transform
img_probabilistic = img.copy()
lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img_probabilistic, (x1, y1), (x2, y2), (0, 255, 0), 1)

# show
images = [img, gray, edges, img_result, img_probabilistic]
colors = [None, "gray", "gray", None, None]
titles = ["img", "gray", "edges", "img_result", "img_probabilistic"]
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], colors[i])
    plt.title(titles[i])
plt.show()
