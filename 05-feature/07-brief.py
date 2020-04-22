import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Paper:
# http://www.gwylab.com/download/BRIEF_2010.pdf
#
# BRIEF algorithm is a descriptor algorithm, not like SIFT/SURF
# 1. BRIEF does not care about how to find potential corners
# 2. BRIEF only contains a descriptor design
# 3. BRIEF can be combined with FAST keypoints result
#
#
# BRIEF essential is SOLVING THE MEMORY PROBLEM
# like the document said SIFT/SURF ..
#
# "Creating such a vector for thousands of features takes a lot of memory
#  which are not feasible for resource-constraint applications
#  especially for embedded systems."
#
# How to save the memory ?
# 1. Replace calculate float number by compare with binary stream Hamming Distance.
# 2. Not generate too complicated descriptor in memory
#
# method1 is also useful in SIFT/SURF process, BUT ...
#
# "we need to find the descriptors first,
#  then only can we apply hashing,
#  which doesn't solve our initial problem on memory."
#
# method2 actually solve the memory problem
# there are 3 main steps refer from official document
#
# 1. selects a set of nd (x,y) location pairs around keypoint in a unique way
# 2. compre the point p,q pixel intensity in each pair, an record the result with 0/1
# 3. take the result binary stream(binary string) as a descriptor
#
# about the "around"
# 1. in paper it define as S x S area
# 2. in openCV which default value S = 31
#
# about the "unique way" detail can be explained in paper, in summary
# 1. mean value sample
# 2. p,q obey same gauss distribution sample
# 3. p,q obey different gauss distribution sample
# 4. use polar coordinate system
# 5. p is fix to (0,0) then q around to p
#
# Orientation Sensitivity:
# From paper we can know
#
# "BRIEF is not designed to be rotationally invariant....
#  it tolerates small amounts of rotation"
#
# from experience if rotate angle should less than 30 degree
#
# IN CONCLUSION:
# BRIEF descriptor not use the data as a descriptor, using the result calculate by data as descriptor
#
#

# read
img = cv.imread("blox.jpg", cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# Initiate FAST detector
star = cv.xfeatures2d.StarDetector_create()

# Initiate BRIEF extractor
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

# find the keypoints with STAR
kp = star.detect(gray, None)

# compute the descriptors with BRIEF
kp, des = brief.compute(gray, kp)

# result
result = img.copy()
cv.drawKeypoints(img, kp, result, (255, 0, 0))

# print size of descriptor, the return value is byte, so it will be 16, 32 and 64
print("brief descriptor size: %s bytes\nbrief descriptor shape %s\n " % (brief.descriptorSize(), des.shape))

# show
plt.subplot(1, 2, 1), plt.imshow(img), plt.title("img")
plt.subplot(1, 2, 2), plt.imshow(result), plt.title("result")
plt.show()
