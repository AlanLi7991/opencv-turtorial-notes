import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Q1: What is Clustering?
# find k labels of a set of data and divide data to k by labels
# like T-shirt size example in document
#
# Q2: Why do we need clustering?
# it gives a label to the raw data
#
# considering in knn, svm training data stage,
# we need some responses to tell us what is data,
# how can we get responses?
# 1. assign data one by one manually
# 2. some algorithm categorizes data set automatically
#
# clustering is the second choice
#
# Q3: Why does k-means sound so familiar?
# in video chapter, we use mean-shift track motive object track.
# actually, mean-shift and k-means clustering have some similar points
#   1. they all calculate the mean value and repeat
#   2. they all need criteria to stop repeat
#   3. they all shit the center in every repeat
#
# Q4: What's the main problem of k-means clustering?
# they are two problems of k-means clustering
#   1. how to choose the category number k
#   2. how to choose the initial centroids of each category
#
#
# Q5: Do we have other clustering algorithms?
# yes, something like
#   * Mean-Shift Clustering
#   * Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
#   * Expectation–Maximization (EM) Clustering using Gaussian Mixture Models (GMM)
#   * Agglomerative Hierarchical Clustering


# random 1-dimension data
x = np.random.randint(25, 100, 25)
y = np.random.randint(175, 255, 25)

# combine data & convert to float32 accuracy
z = np.hstack((x, y))
z = z.reshape((50, 1))
z = np.float32(z)

# define criteria for stop
# ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# set flags
# (Just to avoid line break in the code)
flags = cv.KMEANS_RANDOM_CENTERS

# Apply KMeans
compactness, labels, centers = cv.kmeans(z, 2, None, criteria, 10, flags)

A = z[labels == 0]
B = z[labels == 1]

# show 1d random data
plt.subplot(221)
plt.hist(z, 256, [0, 256])

# plot 'A' in red, 'B' in blue, 'centers' in yellow
plt.subplot(222)
plt.hist(A, 256, [0, 256], color='r')
plt.hist(B, 256, [0, 256], color='b')
plt.hist(centers, 32, [0, 256], color='y')


# random 2-dimension data
X = np.random.randint(25, 50, (25, 2))
Y = np.random.randint(60, 85, (25, 2))

# combine data & convert to np.float32
Z = np.vstack((X, Y))
Z = np.float32(Z)

# define criteria and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Apply KMeans
ret, label, center = cv.kmeans(Z, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

# Now separate the data, Note the flatten()
A = Z[label.ravel() == 0]
B = Z[label.ravel() == 1]

# plot the original 2d data
plt.subplot(223)
plt.scatter(A[:, 0], A[:, 1])
plt.scatter(B[:, 0], B[:, 1], c='r')

# plot the 2d data with center
plt.subplot(224)
plt.scatter(A[:, 0], A[:, 1])
plt.scatter(B[:, 0], B[:, 1], c='r')
plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')

# show
plt.show()
