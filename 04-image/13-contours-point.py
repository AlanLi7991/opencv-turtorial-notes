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

polygon_img = img.copy()
point_red = (50, 50)
point_green = (110, 140)
cv.circle(polygon_img, point_red, 3, (255, 0, 0), -1)
cv.circle(polygon_img, point_green, 3, (0, 255, 0), -1)
test_red = cv.pointPolygonTest(cnt, point_red, True)
test_green = cv.pointPolygonTest(cnt, point_green, True)
print(
    "pointPolygonTest: \n test_red: %s, test_green %s \n"
    % (test_red, test_green)
)

# show
plt.subplot(1, 3, 1), plt.imshow(img, "gray")
plt.subplot(1, 3, 2), plt.imshow(cnt_img, "gray")
plt.subplot(1, 3, 3), plt.imshow(polygon_img, "gray")
plt.show()
