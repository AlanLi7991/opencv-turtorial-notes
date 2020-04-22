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

hull = cv.convexHull(cnt, returnPoints=False)
print("hull: \n", hull, "\n")
defects = cv.convexityDefects(cnt, hull)
print("defects: \n", defects, "\n")

defect_img = img.copy()
#  [ start point, end point, farthest point, approximate distance to farthest point ]
for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv.line(defect_img, start, end, [0, 255, 0], 2)
    cv.circle(defect_img, far, 5, [0, 0, 255], -1)

# show
plt.subplot(2, 2, 1), plt.imshow(img), plt.title("original")
plt.subplot(2, 2, 2), plt.imshow(cnt_img), plt.title("cnt_img")
plt.subplot(2, 2, 3), plt.imshow(defect_img), plt.title("defect_img")
plt.show()
