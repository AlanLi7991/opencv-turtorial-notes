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

# aspect ratio
# the ratio of width to height of bounding rect of the object.
x, y, w, h = cv.boundingRect(cnt)
aspect_ratio = float(w)/h
print("aspect_ratio: \n", aspect_ratio, "\n")

# extend
# the ratio of contour area to bounding rectangle area.
rect_area = w*h
area = cv.contourArea(cnt)
extent = float(area)/rect_area
print("extent: \n", extent, "\n")

# solidity
# the ratio of contour area to its convex hull area.
hull = cv.convexHull(cnt)
hull_area = cv.contourArea(hull)
solidity = float(area)/hull_area
print("solidity: \n", solidity, "\n")


# equivalent diameter
# the diameter of the circle whose area is same as the contour area.
equi_diameter = np.sqrt(4*area/np.pi)
print("equivalent diameter: \n", equi_diameter, "\n")

# orientation
# the angle at which object is directed.
# following method also gives the Major Axis and Minor Axis lengths.
(x, y), (major, minor), orientation = cv.fitEllipse(cnt)
print("orientation: \n", orientation, "\n")
print("(major, minor) axis: \n", major, " ", minor, "\n")


# mask & pixel points
mask = np.zeros(gray.shape, np.uint8)
cv.drawContours(mask, [cnt], 0, 255, -1)
print("mask: \n", mask, "\n")
pixel_points_np = np.transpose(np.nonzero(mask))
pixel_points_cv = cv.findNonZero(mask)
print("pixel points: \n", "use np \n", pixel_points_np, "\nuse cv \n", pixel_points_cv, "\n")


# maximum and minimum's value and locations
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(gray, mask=mask)
print("maximum and minimum: \n", "value: ", min_val, max_val, "locations: ", min_loc, max_loc, "\n")

# mean color or intensity
mean_val = cv.mean(img, mask=mask)
print("mean val: \n", mean_val, "\n")

# extreme points
x_axis = cnt[:, :, 0]
y_axis = cnt[:, :, 1]
min_x = x_axis.argmin()
max_x = x_axis.argmax()
min_y = y_axis.argmin()
max_y = y_axis.argmax()
leftmost = tuple(cnt[min_x][0])
rightmost = tuple(cnt[max_x][0])
topmost = tuple(cnt[min_y][0])
bottommost = tuple(cnt[max_y][0])
print(
    "extreme points: \n leftmost: %s \n rightmost: %s \n topmost: %s\n bottommost: %s \n"
    % (leftmost, rightmost, topmost, bottommost)
)
