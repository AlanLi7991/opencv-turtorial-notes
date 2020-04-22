import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Q1: What is SVM(Support Vector Machines)?
# from the document
#   "So what SVM does is to find a straight line (or hyperplane) with largest minimum distance to the training samples."
# SVM just like KNN, it is also a "classification algorithms"
#
#
# Q2: What problem does SVM want to solve?
#
# save the memory!!!
#
# like document refer( according to the document,)
#   "In kNN, for a test data, we used to measure its distance to
#    all the training samples and take the one with minimum distance.
#    It takes plenty of time to measure all the distances
#    and plenty of memory to store all the training-samples.
#    ......, should we need that much?"
#
# Q3: What concepts in SVM?
#
# Decision Boundary:
#   the imaging line boundary that can separate samples on plane
#
# Linear Separable/Non-Linearly Separable:
#   "Linear Separable" means if all samples locate on a plane, it can be separated by line
#   but if the dimensional of sample data is not 2D, how can we separate it by line?
#   this situation called "Non-Linearly Separable"
#
# Support Vectors:
#   the problem svm want to solve is knn need large memory can save all samples distance,
#   svm only need the samples near the "Decision Boundary",
#   the samples take part in calculating "Decision Boundary" is "Support Vectors"
#   which means "support to calculate"
#
# Support Planes:
#   the imaging lines which plus positive/negative offset with "Decision Boundary",
#   or
#   the lines passing through "Support Vectors"
#   it can improve the classify result accuracy by beyone the planes.
#
#
# Weight Vector/Feature Vector/Bias:
#   "Decision Boundary" is a line, we can present it as ax+by+c = 0
#   or more professional w1x1 + w2x2 + b = 0 => w^Tx + b = 0
#   which
#   w^T = [w1, w2]
#   x = [x1, x2]
#   w is "Weight Vector"
#   x is "Feature Vector"
#   b is "Bias"
#   if sample data dimension not 2D, the length of w,x can be n
#   w = [w1, w2 ... wn], x = [x1, x2 ... xn]
#
# C:
#   a constant value by samples distribution or experience
#   just like the k of KNN, magic number in most of the time
#
# ξ:
#   the error value of misclassification data, from document:
#       "It is the distance from its corresponding training sample to their correct decision region.
#        For those who are not misclassified .... their distance is zero."
#   it means if a sample is correctly classified, the
#       1. classified: ξ = 0
#       2. misclassified: ξ = distance to "Support Planes"
#
#
# Gamma:
#   the parameter γ of a kernel function during decrease dimension to 2D
#
#
# Q4: How to deal with "Non-Linearly Separable" ?
#
# for the data not 2-dimensional which can't be divided into two with a straight line.
# we can just map it to 2D model(!!), so we can separate it by line.
#
# d < 2(one dimension):
#   map it with added dimension, like (x) => (x, x^2)
#
# d > 2(three or higher dimension):
#   decrease the higher dimension to 2-dimension via "kernel function",
#   like the document example
#   attention:
#   !! the document write wrong here, lose pow symbols in line (*)
#
#   2d point:
#       p=(p1,p2), q=(q1,q2).
#   3d point:
#       ϕ(p) = (p21, p22, 2sqrt(p1p2) ), ϕ(q)=(q21, q22, 2sqrt(p1p2) )
#   define a "kernel function" K(p,q)
#   which does a dot product between two 3d points:
#   K(p,q) = ϕ(p).ϕ(q)
#          = (p21, p22, 2sqrt(p1p2) ).(q21, q22, 2sqrt(p1p2) )
# (*)      = (p1q1)^2 + (p2q2)^2 + 2p1q1p2q2
#          = (p1q1+p2q2)^2
#          = (pq)^2
#
#   It means,
#   a dot product in three-dimensional space can be achieved
#   using squared dot to product in two-dimensional space.
#
#
# Q5: What's the main problem of SVM?
#
# How to pick the C value
#
# from document
#   "How ( In which way?)should the parameter C be chosen?
#    It is obvious that the answer to this question depends on
#    how the training data is distributed. (Obviously, the ....)
#    Although there is no general answer."
#
# formula:
#   min L(w,b0) = ||w||^2 + C * ∑(ξ)
#
# * Large values of C:
#   1. less misclassification errors but a smaller margin.
#   2. in this case it is expensive to make misclassification errors.
#   3. since the aim of the optimization is to minimize the argument, few misclassifications errors are allowed.
#
# * Small values of C:
#   1. bigger margin and more classification errors.
#   2. in this case the minimization does not consider that much the term of the sum.
#   3. so it focuses more on finding a hyperplane with big margin.
#
#

# feature set containing (x,y) values of 25 known/training data
train_data = np.random.randint(0, 100, (25, 2)).astype(np.float32)

# labels each one either Red or Blue with numbers 0 and 1
responses = np.random.randint(0, 2, (25, 1))

# take Red families and plot them
red = train_data[responses.ravel() == 0]
# take Blue families and plot them
blue = train_data[responses.ravel() == 1]

# input
newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)

# create svm & configuration
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

# train the data with responses
# make sure the pairs of data
svm.train(train_data, cv.ml.ROW_SAMPLE, responses)

# get svm result
ret, results = svm.predict(newcomer)

# print result & verbose
verbose_results = ["red" if res == 0 else "blue" for res in results[0]]
print("result response {} verbose:  {}\n".format(results, verbose_results))

# show
plt.scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')
plt.scatter(red[:, 0], red[:, 1], 80, 'r', '^')
plt.scatter(blue[:, 0], blue[:, 1], 80, 'b', 's')
plt.show()
