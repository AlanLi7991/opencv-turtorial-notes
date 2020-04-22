import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# read
imgL = cv.imread('tsukuba_l.png', cv.IMREAD_GRAYSCALE)  # left image
imgR = cv.imread('tsukuba_r.png', cv.IMREAD_GRAYSCALE)  # right image

# create stereo based matcher
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)

# calculate disparity
disparity = stereo.compute(imgL, imgR)

# show
plt.imshow(disparity, 'gray')
plt.show()
