import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# TOPIC:
# 1. brute force is a method for matching keypoints with feature
# 2. two important methods are BFMatcher.match() and BFMatcher.knnMatch()
# 3. should use different init method with different parameter
# 4. draw with two methods are cv.drawMatches() and cv.drawMatchesKnn()
# 5. knnMatch parameter returns a list with k size tuple as an item
#

# read
query = cv.imread('box.png', cv.IMREAD_GRAYSCALE)          # queryImage
train = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints and descriptors with ORB
query_kp, query_des = orb.detectAndCompute(query, None)
train_kp, train_des = orb.detectAndCompute(train, None)

# create BFMatcher object with parameter
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(query_des, train_des)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
ret_orb = cv.drawMatches(query, query_kp, train, train_kp, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
query_kp, query_des = sift.detectAndCompute(query, None)
train_kp, train_des = sift.detectAndCompute(train, None)

# create BFMatcher object without parameter
bf = cv.BFMatcher()

# match descriptors with knn(k-nearest neighbors)
matches = bf.knnMatch(query_des, train_des, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
ret_sift = cv.drawMatchesKnn(query, query_kp, train, train_kp, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# show
plt.subplot(2, 2, 1), plt.imshow(query, "gray"), plt.title("query")
plt.subplot(2, 2, 2), plt.imshow(train, "gray"), plt.title("train")
plt.subplot(2, 2, 3), plt.imshow(ret_orb, "gray"), plt.title("ret_orb")
plt.subplot(2, 2, 4), plt.imshow(ret_sift, "gray"), plt.title("ret_sift")
plt.show()
