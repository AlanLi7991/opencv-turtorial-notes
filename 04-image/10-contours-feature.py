import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread('approx.jpg', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# threshold
ret, thresh = cv.threshold(gray, 127, 255, 0)

# find contours
# https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html
# document has error
# findContours had not changed source img and plus one return value at first
some, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# get first
cnt = contours[0]
cnt_img = img.copy()
cv.drawContours(cnt_img, contours, 0, (255, 0, 0), 3)
# moments
M = cv.moments(cnt)
print("moments: \n", M, "\n")

# read centroid
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print("centroid: \n", "cx: ", cx, " cy: ", cy, "\n")

# area
area = cv.contourArea(cnt)
print("area: \n", area, "\n")

# perimeter
perimeter = cv.arcLength(cnt, True)
print("perimeter: \n", perimeter, "\n")

# approximation
epsilon_01 = 0.03*cv.arcLength(cnt, True)
epsilon_001 = 0.01*cv.arcLength(cnt, True)
approx_01 = cv.approxPolyDP(cnt, epsilon_01, True)
approx_001 = cv.approxPolyDP(cnt, epsilon_001, True)

# draw approximation
approx_01_img = img.copy()
approx_001_img = img.copy()
cv.drawContours(approx_01_img, [approx_01], 0, (255, 0, 0), 3)
cv.drawContours(approx_001_img, [approx_001], 0, (255, 0, 0), 3)


# hull
hull = cv.convexHull(cnt)
print("hull: \n", hull, "\n")
hull_img = img.copy()
cv.drawContours(hull_img, [hull], 0, (255, 0, 0), 3)

# convexity
convexity = cv.isContourConvex(cnt)
print("convexity: \n", convexity, "\n")

# bounding rectangle
bounding = cv.boundingRect(cnt)
print("bounding: \n", bounding, "\n")
minArea = cv.minAreaRect(cnt)
print("minArea: \n", minArea, "\n")
circle = cv.minEnclosingCircle(cnt)
print("circle: \n", circle, "\n")
ellipse = cv.fitEllipse(cnt)
print("ellipse: \n", ellipse, "\n")
line = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
print("line: \n", line, "\n")

shape_img = img.copy()

x, y, w, h = bounding
cv.rectangle(shape_img, (x, y), (x+w, y+h), (255, 0, 0))

box = cv.boxPoints(minArea)

center = (int(circle[0][0]), int(circle[0][1]))
radius = int(circle[1])
cv.circle(shape_img, center, radius, (0, 255, 0))

cv.ellipse(shape_img, ellipse, (0, 0, 255))

# show
images = [cnt_img, approx_01_img, approx_001_img, hull_img, shape_img]
titles = ["cnt_img", "approx_01_img", "approx_001_img", "hull_img", "shape_img"]
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
plt.show()
print("end")
