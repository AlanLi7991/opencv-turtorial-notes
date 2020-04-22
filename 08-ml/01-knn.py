import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Q1: What is kNN(k-Nearest Neighbor)?
# from the document
#   "kNN is one of the simplest classification algorithms available for supervised learning."
#
# Q2: What problem does KNN want to solve?
# "classification algorithms" is the key point
# take one unclassified input data in to some predefined classes is a classification
#
# Q3: What concepts in KNN?
# train data:
#   because kNN need predefined pairs during the algorithms,
#   the data making pairs is training data
#
# responses:
#   responses are training data classified results
#   responses size should be equal to train data
#
# test data:
#   a list of input data,
#   each input data should be classified as one result of responses
#
# labels:
#   the result of test data,
#   same as responses, but with a different name
#
# distance:
#   the number for judging neighbors,
#   k nearest denote k minimums
#
# Q4: What's the main problem with kNN?
# from document
# "But there is a problem with that. Red Triangle may be the nearest. But what if there are a lot of Blue Squares near to him? "
# if the number of red/blue are equal, the location of red/blue distribute maybe not, is it import?
#
# 1. how to choose the import k?
# 2. we are supposing all k neighbors are with equal importance? Is it justice?
#
#

# Feature set containing (x,y) values of 25 known/training data
trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)

# Labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

# Take Red families and plot them
red = trainData[responses.ravel() == 0]
# Take Blue families and plot them
blue = trainData[responses.ravel() == 1]

# input
newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)

# create knn
knn = cv.ml.KNearest_create()

# train the data with responses
# make sure the pairs of data
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)

# get knn result
ret, results, neighbours, dist = knn.findNearest(newcomer, 3)

# print result & verbose
verbose_results = ["red" if res == 0.0 else "blue" for res in results[0]]
verbose_neighbours = ["red" if res == 0.0 else "blue" for res in neighbours[0]]
print("result response {} verbose:  {}\n".format(results, verbose_results))
print("neighbours response {} verbose:  {}\n".format(neighbours, verbose_neighbours))
print("distance from neighborhood:  {}\n".format(dist))

# show
plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')
plt.show()
