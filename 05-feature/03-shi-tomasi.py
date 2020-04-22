import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# read
img = cv.imread("blox.jpg", cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# find shi-tomasi corners
# parameter (25): maxCorners , if <= 0 implies that no limit
# parameter (0.01): qualityLevel , means the accpet level of measure, like harris's dst > 0.01*dst.max() operate
# parameter (10): minDistance, the minimum measure distance of two corners
corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)

# draw
for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 5, (0, 255, 0), 1)

plt.imshow(img), plt.show()
