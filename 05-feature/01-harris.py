import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread("blox.jpg", cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
float_gray = np.float32(gray)

# harris
# blockSize: use 2x2 neighborhood pixels to calculate harris
# ksize: the sobel operator size, to calculate x,y edges, 3 means 3x3 kernel
# k: the response function constant, the empirical value is 0.04~0.06
dst = cv.cornerHarris(float_gray, 2, 3, 0.04)

# result is dilated for marking the corners, not important
dst = cv.dilate(dst, None)

# threshold for an optimal value, it may vary depending on the image.
result = img.copy()
result[dst > 0.01*dst.max()] = [255, 0, 0]

# make threshold result more accurate
accurate = img.copy()
mask = np.where(dst > 0.1 * dst.max(), 1, 0)
index = mask > 0
accurate[index] = [0, 255, 0]

# show
plt.subplot(2, 2, 1), plt.imshow(img), plt.title("img")
plt.subplot(2, 2, 2), plt.imshow(result), plt.title("result")
plt.subplot(2, 2, 3), plt.imshow(accurate), plt.title("accurate")
plt.show()
