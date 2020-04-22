import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread('match_shapes.jpg', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# threshold
ret, thresh = cv.threshold(gray, 127, 255, 0)

# find contours
# https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html
# document has error
# findContours had not changed source img and plus one return value at first
some, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
ret_img = cv.cvtColor(some, cv.COLOR_GRAY2RGB)

n = len(contours)
# draw all contours
cnt_img = img.copy()
for i in range(n):
    cnt = contours[i]
    cv.drawContours(cnt_img, contours, i, (255, 0, 0), 2)

# draw match contours
match_img = img.copy()
results = []
for i in range(n):
    for j in range(n-1, i, -1):
        match = cv.matchShapes(contours[i], contours[j], cv.CONTOURS_MATCH_I1, 0.0)
        results.append((i, j, match))
results = np.array(results)
min_match = results[:, 2].argmin()
i, j = results[min_match][0:2]
cv.drawContours(match_img, contours, int(i), (0, 255, 0), 2)
cv.drawContours(match_img, contours, int(j), (0, 255, 0), 2)

# show
plt.subplot(2, 2, 1), plt.imshow(img), plt.title("img")
plt.subplot(2, 2, 2), plt.imshow(cnt_img), plt.title("cnt_img")
plt.subplot(2, 2, 3), plt.imshow(ret_img), plt.title("ret_img")
plt.subplot(2, 2, 4), plt.imshow(match_img), plt.title("match_img")
plt.show()
