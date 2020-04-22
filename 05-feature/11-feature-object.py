import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# const define
MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1

# read
query = cv.imread('box.png', cv.IMREAD_GRAYSCALE)          # queryImage
train = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE)  # trainImage

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
query_kp, query_des = sift.detectAndCompute(query, None)
train_kp, train_des = sift.detectAndCompute(train, None)

# create flann mather
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# find knn matches
matches = flann.knnMatch(query_des, train_des, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# check good
if len(good) > MIN_MATCH_COUNT:
    # get good points of query & train
    # here class is list
    query_pts = [query_kp[m.queryIdx].pt for m in good]
    train_pts = [train_kp[m.trainIdx].pt for m in good]

    # convert float & reshape points
    # make list class to np array
    query_pts = np.float32(query_pts).reshape(-1, 1, 2)
    train_pts = np.float32(train_pts).reshape(-1, 1, 2)

    # find transform
    # here use RANSAC (Random Sample Consensus)
    # will remove noise(outlier) points
    # return value
    # 1. matrix M
    # 2. mask for the data(inlier) points take part in calculate M
    M, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)

    # mask is np array with (73, 1, 1) shape
    # here convert np array to list
    matchesMask = mask.ravel().tolist()

    # get size of query image
    height, width = query.shape

    # create bounds point of query image
    # ([left-top], [left-bottom], [right-bottom], [right-top])
    # then reshape to np array
    bounds = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)

    # perspective transform bounds of query to location of train via M
    location = cv.perspectiveTransform(bounds, M)

    # type location to int32
    location = np.int32(location)

    # draw white polylines on train image
    obj = train.copy()
    cv.polylines(obj, [location], True, 255, 3, cv.LINE_AA)
else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

# prepare draw matches parameters
draw_params = dict(
    matchColor=(0, 255, 0),  # draw matches in green color
    singlePointColor=None,
    matchesMask=matchesMask,  # draw only inliers
    flags=2
)

# draw only good matches
result = cv.drawMatches(query, query_kp, train, train_kp, good, None, **draw_params)

# show
plt.subplot(2, 2, 1), plt.imshow(query, "gray"), plt.title("query")
plt.subplot(2, 2, 2), plt.imshow(train, "gray"), plt.title("train")
plt.subplot(2, 2, 3), plt.imshow(obj, "gray"), plt.title("object locate in train")
plt.subplot(2, 2, 4), plt.imshow(result, "gray"), plt.title("result with matches")
plt.show()
