import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread('lena.tif', cv.IMREAD_GRAYSCALE)

# global threshold
ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

# adaptive threshold
thresh6 = cv.adaptiveThreshold(
    img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
thresh7 = cv.adaptiveThreshold(
    img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

# Otsu's thresholding
ret, thresh8 = cv.threshold(img, 0, 255, cv.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img, (5, 5), 0)
ret, thresh9 = cv.threshold(blur, 0, 255, cv.THRESH_OTSU)

# show
images = [
    img, thresh1, thresh2, thresh3, thresh4,
    thresh5, thresh6, thresh7, thresh8, thresh9
]
colors = [
    "gray", "gray", "gray", "gray", "gray",
    "gray", "gray", "gray", "gray", "gray"
]
titles = [
    "image", "THRESH_BINARY", "THRESH_BINARY_INV",
    "THRESH_TRUNC", "THRESH_TOZERO", "THRESH_TOZERO_INV",
    "ADAPTIVE_THRESH_MEAN_C", "ADAPTIVE_THRESH_GAUSSIAN_C",
    "THRESH_OTSU", "THRESH_OTSU+GaussianBlur"
]
for i in range(len(images)):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i], colors[i])
    plt.title(titles[i])
plt.show()
