import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# read img
img_raw = cv.imread('mario.jpg', cv.IMREAD_COLOR)
img_rgb = cv.cvtColor(img_raw, cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img_raw, cv.COLOR_BGR2GRAY)
template = cv.imread('mario_coin.jpg', cv.IMREAD_GRAYSCALE)

# match
res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)

# take threshold
threshold = 0.8

# convert a bool matrix use threshold
condition = res >= threshold

# get true value location
# first element is row number which mean height index
# second element is column number which mean width index
# it denote (height index list, width index list)
loc = np.where(condition)

# reverse location to (width index, height index) and zip to a tuple
points = tuple(zip(*loc[::-1]))

# get template size
w, h = template.shape[::-1]

# draw rect
for pt in points:
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

# show
plt.subplot(1, 3, 1), plt.imshow(img_gray, "gray"), plt.title("Gray")
plt.subplot(1, 3, 2), plt.imshow(res, "gray"), plt.title("Match")
plt.subplot(1, 3, 3), plt.imshow(img_rgb, None), plt.title("Result")
plt.show()
