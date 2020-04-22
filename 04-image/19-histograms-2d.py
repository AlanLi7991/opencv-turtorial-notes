import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread('lena.tif', cv.IMREAD_COLOR)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 2d histogram cv
hist_cv = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# 2d histogram numpy
h = hsv[:, :, 0]
s = hsv[:, :, 1]
hist_np, xbins, ybins = np.histogram2d(h.ravel(), s.ravel(), [180, 256], [[0, 180], [0, 256]])


# the result we get is a two dimensional array of size 180x256.
# It will be a grayscale image and it won't give much idea what colors are there
plt.subplot(1, 2, 1), plt.imshow(hist_cv, "gray"), plt.title("hist_cv")
plt.subplot(1, 2, 2), plt.imshow(hist_np, "gray"), plt.title("hist_np")
plt.show()
