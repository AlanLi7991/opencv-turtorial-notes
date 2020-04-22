import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# SURF algorithm's target is speeding up SIFT via 4 methods
#
# METHOD1 : Use Integral-Image(积分图像) replace DoG
# METHOD2 : Use box filter replace resize layer in octave
# METHOD3 : Use Haar-Wavelet replace histogram during orientation block
# METHOD4 : Reduce descriptor dimension, using flag to give up rotation invariance
#
# here we can get more information from source code
# https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/surf.cpp
#
#
# Here we pickup some essential verbose of the official document
#
# Essential 1:
# " "SURF: Speeded Up Robust Features" introduced a new algorithm called SURF.
# As name suggests, it is a speeded-up version of SIFT."
#
# Essential 2:
# "In SIFT, Lowe approximated Laplacian of Gaussian with Difference of Gaussian for finding scale-space.
# SURF goes a little further and approximates LoG with Box Filter"
#
# Essential 3:
# "For many applications, rotation invariance is not required,
# so no need of finding this orientation, which will speed up the process."
#
#
# Then compare SURF with SIFT step by step
#
# 1. Prepare images & search extrema
# 1-1. Prepare Integral-Image
#      Source Code Hint: SURF_Impl::detectAndCompute()
# Compare With SIFT:
# same:
# None, SIFT does not contain this step
#
# differ:
# SIFT will first resize the input image to multiples, then apply gauss blur
# SURF use "integral(img, sum, CV_32S);" at line 938 get Integral-Image at first
# then use fastHessianDetector() create Prymaids & Octaves
#
# 1-2. Fill up Prymaids & Octaves
#      Source Code Hint: fastHessianDetector() & calcLayerDetAndTrace()
# Compare With SIFT:
# same:
# turn single input image to Prymaids count * Octaves count images
#
# differ:
# In SIFT the size of different octaves not equal
# In SURF the size of each images equals to original image size
# SIFT use gauss blur (differnet sigma) & substract to get DoG
# SURF use Hessian Matrix (differnet step) as kernel do convolution
#
# Speed Up Reason: (METHOD1 METHOD2)
# Hessian Matrix computer is faster than gauss blur during convolute
# caused by the help of Integral-Image
#
# 1-3. Find extrema point
#      Source Code Hint: SURFFindInvoker::findMaximaInLayer()
# Compare With SIFT: almost same, use 26 pixels around in space
#
#
# 2. Keypoint Localization
# 2-1. Non-maxima suppression
#      Source Code Hint: SURFFindInvoker::findMaximaInLayer() - Line417
# Compare With SIFT:
# same:
# just like the SIFT refines the potential points to keypoint
#
# differ:
# SIFT need to flatten the keypoints to origin coordinate
# SURF did not resize the images, alternate with different step box filter
# no need the flatten step
#
# 3. Orientation Assignment
# 3-1. Statistics wavelet value in 6 area of 360 degree
#      Source Code Hint: SURFInvoker() + ORI_RADIUS
# Compare With SIFT:
# same:
# this step aims for making sure the orientation
#
# differ:
# SIFT use histogram with 36bins
# SURF use the sum of wavelet value of area
# SIFT has a main orientation
# and multiple associate orientations with 80% threshold
# SURF just precision in a radius area with max sum value in 6bins
# SURF can use upright flag for giving up this step
#
# Speed Up Reason: (METHOD3)
# addition runs faster than histogram compare
# flag can skip this step
#
# 4. Keypoint Descriptor
# 4-1. Neighborhood pixels & Rotate
#      Source Code Hint: SURFInvoker()-operator()-PATCH_SZ
# Compare With SIFT:
# same:
# select an area around the keypoint, then divide to block
#
# differ:
# SIFT select fix area by pixel size 16 * 16 , block size is 4 * 4
# SURF use dynamic area by window via "int win_size = (int)((PATCH_SZ+1)*s);"
# then divide the window area to 4 * 4 = 16 sub-region (same as block)
# not depend on the pixels number
#
#
# 4-2. Represent sub-region with 4 length vector
#      Source Code Hint: SURFInvoker()-operator()-Line780
# Compare With SIFT:
# same:
# each block/sub-region has own vector
#
# differ:
# SIFT use a small histogram with 8bins representing block
# SURF use a sub region accumulate v=(∑dx,∑dy,∑|dx|,∑|dy|)
# SURF can use flag extend the vector size to 8
# by divide the dx and |dx| in dy<0 and dy≥0
#
# 4-3. Assembly to descriptor
#      Source Code Hint: SURFInvoker()-operator()-Line850
# Compare With SIFT:
# same:
# descriptor consist of block/sub-region vector
#
# differ:
# SIFT descriptor size is fix 16 sub 4*4 block * 8bins = 128
# SURF can use flag switch between
# 16 sub-region * 4 vector = 64
# and
# 16 sub-region * 8 vector divide by zero = 128
#
# Speed Up Reason: (METHOD4)
# addition of intergal-image value run faster
#
#
# Refer:
# https://zzzzzch.github.io/2017/12/23/Surf/
# https://www.cnblogs.com/gfgwxw/p/9415218.html
#

# read
img = cv.imread('butterfly.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# Create SURF object. You can specify params here or later.
# Here I set Hessian Threshold to 400
surf = cv.xfeatures2d.SURF_create(400)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(gray, None)
low = img.copy()
cv.drawKeypoints(gray, kp, low, (255, 0, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# We set it to some 50000. Remember, it is just for representing in picture.
# In actual cases, it is better to have a value 300-500
surf.setHessianThreshold(50000)
kp, des = surf.detectAndCompute(gray, None)
high = img.copy()
cv.drawKeypoints(gray, kp, high, (0, 255, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# Check upright flag, if it False, set it to True
surf.setUpright(True)
kp, des = surf.detectAndCompute(gray, None)
upright = img.copy()
cv.drawKeypoints(gray, kp, upright, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# So we make it to True to get 128-dim descriptors.
surf.setExtended(True)
kp, des = surf.detectAndCompute(gray, None)
extend = img.copy()
cv.drawKeypoints(gray, kp, extend, (128, 128, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# show
plt.subplot(3, 2, 1), plt.imshow(img), plt.title("img")
plt.subplot(3, 2, 2), plt.imshow(low), plt.title("low")
plt.subplot(3, 2, 3), plt.imshow(high), plt.title("high")
plt.subplot(3, 2, 4), plt.imshow(upright), plt.title("upright")
plt.subplot(3, 2, 6), plt.imshow(extend), plt.title("extend")
plt.show()
