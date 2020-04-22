import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#
#
# Q1: What problem does epipolar geometry want to solve
# use more than one camera to find the depth information loosen by taking an image using a pin-hole camera
#
# Q2: How to understand epipolar means
# the prefix "epi" from Greek epi ‘upon, near to, in addition’. which denote the concept "space around"
# "polar" from the word "pole"
# "pole" means "a long, slender, rounded piece of wood or metal
# typically used with one end placed in the ground as a support for something"
# so the "polar" in geometry field is explained in
# "the straight line joining the two points at which tangents from a fixed point touch a conic section."
# the final explanation of "epipolar geometry" is
# the geometry use the thin line(polar) to find the space information(epi-)
#
# Q3: What concepts in this algorithm
# EPILINE:
#   document refers
#   "The projection of the different points on OX form a line on the right plane (line l′)."
#   the point is the projection on the right image, it is pixels on image coordinate
#   epiline corresponding to the point x on the left image
#   it can be described as "epipolar constraint"
#
# EPIPOLAR CONSTRAINT:
# from the document
# "It means, to find the point x (correspond pixel location) on the right image,
# search along this epiline. It should be somewhere on this line"
# the document adds
# "Think of it this way,
# to find the matching point in other images,
# you need not to search the whole image,
# just search along the epiline.
# So it provides better performance and accuracy"
#
# EPIPOLAR PLANE:
# all points will have its corresponding epilines in the other image.
# look at the image on the document to understand
#
# EPIPOLE:
# right camera projection pixel location on left image called "epipole"
#
# Q4: How to find epipolar lines and epipoles above?
# to find them, we need two more ingredients, Fundamental Matrix (F) and Essential Matrix (E).
#
#
# Q5: The difference between Fundamental & Essential
# Essential Matrix :
# contains information about translation and rotation,
# which describes the location of the second camera
# relative to the first in global coordinates.
#
# Fundamental Matrix :
#   contains the same information as Essential Matrix in addition
#   to the information about the intricacies of both cameras
#   so that we can relate the two cameras in pixel coordinates.
#   (If we are using rectified images and normalize the point by dividing by the focal lengths, F=E).
#
# because Essential is a subset of Fundamental, we can say
# Fundamental Matrix F maps a point in one image to a line (epiline) in the other image.
#

# draw epilines
def drawlines(canvas, data, lines, pts_l, pts_r):
    ''' canvas - image on which we draw the epilines for the points in data
        lines - corresponding epilines '''
    row, column = canvas.shape
    canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)
    data = cv.cvtColor(data, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts_l, pts_r):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [column, -(r[2]+r[0]*column)/r[1]])
        canvas = cv.line(canvas, (x0, y0), (x1, y1), color, 1)
        canvas = cv.circle(canvas, tuple(pt1), 5, color, -1)
        data = cv.circle(data, tuple(pt2), 5, color, -1)
    return canvas, data

# read
left = cv.imread('myleft.jpg', cv.IMREAD_GRAYSCALE)   # left image
right = cv.imread('myright.jpg', cv.IMREAD_GRAYSCALE)  # right image

# find the keypoints and descriptors with SIFT
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(left, None)
kp2, des2 = sift.detectAndCompute(right, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
good = []
pts_l = []
pts_r = []

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts_l.append(kp1[m.queryIdx].pt)
        pts_r.append(kp2[m.trainIdx].pt)

pts_l = np.int32(pts_l)
pts_r = np.int32(pts_r)
# calculates a fundamental matrix from the corresponding points in two images.
# where F is a fundamental matrix
# mask is for the points take part in calculates after sample
F, mask = cv.findFundamentalMat(pts_l, pts_r, cv.FM_LMEDS)

# We select only inlier points
pts_l = pts_l[mask.ravel() == 1]
pts_r = pts_r[mask.ravel() == 1]

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts_r.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
ret_l, data_r = drawlines(left, right, lines1, pts_l, pts_r)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts_l.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
ret_r, data_l = drawlines(right, left, lines2, pts_r, pts_l)

# show
plt.subplot(221), plt.imshow(ret_l)
plt.subplot(222), plt.imshow(data_r)
plt.subplot(223), plt.imshow(ret_r)
plt.subplot(224), plt.imshow(data_l)
plt.show()
