import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Paper:
# http://dev.ipol.im/~reyotero/bib/bib_all/2008_Roster_Porter_Drummond_faster_better_corner_detection.pdf
#
# FAST algorithm not like SIFT/SURF in two points
# 1. FAST is focusing on looking at potential corners, fast enough in real-time
# 2. FAST does not contain a descriptor design like SIFT
#
# FAST algorithm has two versions in 2006 and 2010
# 2006: it just gives a proposal on how to search a corner point
# 2010: it improves the accuracy with machine learning
#
#
# HERE WE ILLUSTRATE SOME IMPORT SENTENCE IN THE ARTICLE
#
# SENTENCE 1:
#
# From the article, we know the essential definition:
#
# "The original detector classifies p as a corner
# if there exists a set of n contiguous pixels in the circle
# which are all brighter than the intensity of
# the candidate pixel Ip plus a threshold t, or all darker than Ip − t"
#
# comment:
# n: n = 12 in this case
# Ip: Intensity of Pixel(Point) what means a gray value
# t: threshold, filter is [0, Ip - t) & (Ip + t, 255]
#
# SENTENCE 2:
#
# Now the problem convert to "how to detect the contiguous pixels fast"
# answer is detecting with diagonal points of the 16 pixels
# the article said:
#
# "The high-speed test examines pixels 1 and 9.
# If both of these are within t if Ip,
# then p can not be a corner.
# If p can still be a corner, pixels 5 and 13 are examined.
# If p is a corner then
# at least three of these must all be brighter than Ip + t
# or darker than Ip − t.
# If neither of these is the case, then p cannot be a corner."
#
#
# BUT THERE ARE SOME WEAKNESSES
#
# if more details, please read the article & document
# 1. not good when n != 12 (article)
# 2. not optimal (article)
# 3. data waste (document)
# 4. too close (article)
#
# HOW TO SOLVE THESE WEAKNESSES
#
# 1,2,3: use machine learning ID3 algorithm to create a decision tree
# 4: use the nonmax suppression to refine the result
#
# ID3 algorithm:
# 1. train from some images contain keypoint
# 2. create a table of point1, intensity, a new bool value indict if it is a keypoint
# 3. use the formula Hg = H(P) − H(Pd) − H(Ps) − H(Pb) calculate the gain
# 4. once the "decision tree" is created, can be used in FAST
#
# nonmax suppression:
# sum the area intensity around every p, then keep the max in a distance
#
#

# read
img = cv.imread("blox.jpg", cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(gray, None)
result = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

# Print all default params
print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

# Disable nonmaxSuppression
fast.setNonmaxSuppression(False)
kp = fast.detect(gray, None)
disable = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))

# show
plt.subplot(2, 2, 1), plt.imshow(img), plt.title("img")
plt.subplot(2, 2, 2), plt.imshow(result), plt.title("Enable NonmaxSuppression")
plt.subplot(2, 2, 3), plt.imshow(disable), plt.title("Disable NonmaxSuppression")
plt.show()
