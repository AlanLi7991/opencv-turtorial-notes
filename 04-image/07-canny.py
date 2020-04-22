import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Q1: What is "Canny Edge Detection"?
# "Canny Edge Detection" is an algorithm for extracting edge by gradient
#
# Q2: What could Canny do during edge detecting?
# 1. denoising with a 5x5 gaussian filter
# 2. calculating intensity gradient by Sobel kernel in both horizontal and vertical
# 3. using "Non-maximum Suppression" to get maximum value locally
# 4. using "Hysteresis Thresholding" to judge the pixel when not so confirmed it is the edge.
#
# Q3: Why Canny is more accurate than a simple gradient?
# "Non-maximum Suppression":
# filter data by local comparing, getting the maximum of a window
#
# "Hysteresis Thresholding":
# select two thresholds, the maximum and the minimum
#
# the value which is bigger than max must be the edge,
# the value which is smaller than min must not be the edge
#
# other pixels vary its value between the min and the max,
# check the type of pixel it connects to,
# if edge pixel(bigger than max), it is an edge,
# if not edge pixel(smaller than the minimum), it is not (vice versa)
#
#
# Q4: What does the word "Hysteresis" mean?
# the phenomenon in which the value of a physical property lags behind changes in the effect causing it.
#

# read
img = cv.imread('lena.tif', cv.IMREAD_GRAYSCALE)

# canny
canny = cv.Canny(img, 100, 200)

# show
plt.subplot(1, 2, 1), plt.imshow(img, "gray")
plt.subplot(1, 2, 2), plt.imshow(canny, "gray")
plt.show()
