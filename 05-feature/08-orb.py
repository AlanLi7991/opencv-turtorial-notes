import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Paper:
# http://citeseerx.ist.psu.edu/viewdoc/download?doi = 10.1.1.370.4395 & rep = rep1 & type = pdf
#
# ORB algorithm full name is "Oriented FAST and Rotated BRIEF"
# which means
# 1. ORB use detector like FAST to find potential corners, the document said "basically a fusion of FAST"
# 2. ORB use descriptors like BRIEf to describe keypoints, the document said "rBRIEF"
# 3. FAST algorithm isn't "rotate invariant", ORB add orientation to it
# 4. BRIEF descriptor isn't as good as original after rotate, ORB use greedy search to solve it
#
# Q1: How to assign FAST algorithm an orientation?
# use "intensity centroid" point and "center" point as the vector direction
# use image moments(图像矩) calculate the "intensity centroid", refer to 03-contours-feature
#
# In order to get rotate invariant
# ORB must rotate BRIEF to domain orientation(called steer in the document)
# but BRIEF will lose accurate after rotating.
#
# Q2: BRIEF is rotation sensitivity, why?
# BRIEF has a large variance and a mean near 0.5
# if it rotated to domain orientation,
# the points in the image left & right corner
# that relative to domain orientation will look similar
# then the variance will decrease
#
# Q3: How to promote the rotated(steered) BRIEF result?
# in paper use leaning method base on PASCAL 2006 set data
#
# 1. Run each test against all training patches.
# 2. Order the tests by their distance from a mean of 0.5, forming the vector T.
# 3. Greedy search
#
# then the document said
# "a greedy search among all possible binary tests
# to find the ones that have both
# high variance and means close to 0.5"
#
# get the final result rBRIEF
#
# Q4: How about the scale-invariant of ORB?
# from the paper it talks a little about this
#
# "FAST does not produce multi-scale features.
# We employ a scale pyramid of the image,
# and produce FAST features(filtered by Harris)
# at each level in the pyramid."
#
# but no detail of this chapter
#
# IN CONCLUSION:
# ORB algorithm add orientation to FAST, use Harris filtered in pyramids and solve the rotation sensitivity of BRIEF
#

# read
img = cv.imread("blox.jpg", cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(gray, None)

# compute the descriptors with ORB
kp, des = orb.compute(gray, kp)

# draw only keypoints location,not size and orientation
result = img.copy()
result = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

# show
plt.subplot(1, 2, 1), plt.imshow(img), plt.title("img")
plt.subplot(1, 2, 2), plt.imshow(result), plt.title("result")
plt.show()
