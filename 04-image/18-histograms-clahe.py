import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#
# Q: What is CLAHE?
# Contrast Limited Adaptive Histogram Equalization
#
#

# read
img = cv.imread('lena.tif', cv.IMREAD_GRAYSCALE)

# equalize with cv
img_equal = cv.equalizeHist(img)

# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img)

# show
plt.subplot(1, 3, 1), plt.imshow(img, "gray"), plt.title("original")
plt.subplot(1, 3, 2), plt.imshow(img_equal, "gray"), plt.title("equal")
plt.subplot(1, 3, 3), plt.imshow(img_clahe, "gray"), plt.title("clahe")
plt.show()
