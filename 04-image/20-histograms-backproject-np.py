import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Here is the back project algorithm implementing by numpy
# "Back" means from histogram to image
#
# Topic:
# 1. project roi&image to a histogram
# 2. process histogram, maybe divide
# 3. back project histogram to image to get intension image
# 4. filter & threshold intension image get a mask
# 5. use mask to get roi
#

# roi is the object or region of object we need to find
roi = cv.imread('rose_red.png')
roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

# target is the image we search in
img = cv.imread('rose_img.png')
img_origin = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Find the histograms using calcHist. Can be done with np.histogram2d also
roi_hist = cv.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
img_hist = cv.calcHist([img_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

# 0 intension dictate none of this hue & saturate exist on image
# replace 0 to 255, render as white for discriminate it
img_m = np.ma.masked_equal(roi_hist, 0)
img_m = np.ma.filled(img_m, 255).astype('uint8')
img_i = np.ma.masked_equal(img_hist, 0)
img_i = np.ma.filled(img_i, 255).astype('uint8')

# create R = M/I
# if some hue exist a little or not exist in M(roi)
# the result of R will be little or zero
# this denote substract color not in roi
ratio_hist = roi_hist/img_hist

# split image hue saturate
h, s, v = cv.split(img_hsv)

# fancy index with ratio_hist
# if first pixel (x, y) has value (50, 175, 80)
# hue = 50, saturate = 175
# look for it from ration histogram
# if it exist means this hue, saturate is one of roi
# then fill the (x, y) of "backproject image" use intension
#
# the return value shape is a list length equal to ravel()
img_bp = ratio_hist[h.ravel(), s.ravel()]

# threshold with 0 ~ 1
# Q1: why ratio_hist some coordinate can bigger than 1?
# if roi not a subset of image
# Q1: why ratio_hist some coordinate can be NaN?
# none of roi or image contain this hsv
img_bp = np.minimum(img_bp, 1)

# reshape list to size of image
img_bp = img_bp.reshape(img_hsv.shape[:2])

# show gray image redrawn by ratio_hist
img_r = img_bp.copy()

# create kernel
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# filter convolution, for denoise
img_bp = cv.filter2D(img_bp, -1, disc)

# denoise result redrawn by ratio_hist
img_con = img_bp.copy()

# convert int
img_bp = np.uint8(img_bp)

# normalize
cv.normalize(img_bp, img_bp, 0, 255, cv.NORM_MINMAX)

# threshold
ret, thresh = cv.threshold(img_bp, 200, 255, 0)

# create mask
mask = cv.merge((thresh, thresh, thresh))

# biswise and
res = cv.bitwise_and(img, mask)

# show result
img_res = cv.cvtColor(res, cv.COLOR_BGR2RGB)

# show
images = [
    img_origin, img_m, img_i, ratio_hist,
    img_r, img_con, mask, img_res
]
colors = [
    None, "gray", "gray", "gray",
    "gray", "gray", "gray", None
]
titles = [
    "original", "roi_hist", "img_hist", "ratio_hist",
    "ratio_hist redraw", "convolution", "mask", "result"
]
for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i], colors[i])
    plt.title(titles[i])
plt.show()
