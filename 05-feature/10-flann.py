import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# FLANN:  Fast Library for Approximate Nearest Neighbors
#
# TOPIC:
# 1. init method with two dictionary, index/search
# 2. dictionary key refer document
#    https://docs.opencv.org/2.4/modules/flann/doc/flann_fast_approximate_nearest_neighbor_search.html
# 3. use matches mask when draw matches
# 4. [1, 0] means use first element of each tuple in return list
#
#
#

# read
query = cv.imread('box.png', cv.IMREAD_GRAYSCALE)          # queryImage
train = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
query_kp, query_des = sift.detectAndCompute(query, None)
train_kp, train_des = sift.detectAndCompute(train, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)
cv.FlannBasedMatcher_create()

matches = flann.knnMatch(query_des, train_des, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matchesMask,
    flags=cv.DrawMatchesFlags_DEFAULT
)

ret_sift = cv.drawMatchesKnn(query, query_kp, train, train_kp, matches, None, **draw_params)

plt.imshow(ret_sift), plt.show()
