import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# roi is the object or region of object we need to find
roi = cv.imread('rose_red.png')
hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

# train is the image we search in
train = cv.imread('rose_img.png')
hsvt = cv.cvtColor(train, cv.COLOR_BGR2HSV)

# calculating object histogram
roi_hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
img_hist = np.ma.masked_equal(roi_hist, 0)
img_hist = np.ma.filled(img_hist, 255).astype("uint8")

# normalize histogram
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# apply backprojection
dst = cv.calcBackProject([hsvt], [0, 1], roi_hist, [0, 180, 0, 256], 1)

# convolute with circular disc
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# filter convolution, for denoise
cv.filter2D(dst, -1, disc, dst)

# threshold
ret, thresh = cv.threshold(dst, 200, 255, 0)

# create mask
mask = cv.merge((thresh, thresh, thresh))

# biswise and
res = cv.bitwise_and(train, mask)

# show
plt.subplot(2, 2, 1), plt.imshow(train), plt.title("train")
plt.subplot(2, 2, 2), plt.imshow(img_hist, "gray"), plt.title("img_hist")
plt.subplot(2, 2, 3), plt.imshow(mask, "gray"), plt.title("mask")
plt.subplot(2, 2, 4), plt.imshow(res), plt.title("result")
plt.show()
