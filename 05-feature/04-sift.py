import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread('home.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# create sift instance
# comment phase of the algorithm for official document
# here can get more information from source code
# https://github.com/opencv/opencv_contrib/blob/master/modules/xfeatures2d/src/sift.cpp
#
# 1. Scale-space Extrema Detection
# 1-1. Pyramids & Octaves
# Source Code Hint: buildGaussianPyramid()
# process the input image to a scale Pyramids
# then in each scale layer, multiple the image with GaussBlur in different sigma
# the group of the same scale including different GaussBlur result is called "octave"
# in this step algorithm transform a single image to Pyramids layer count * Octaves count images
#
# 1-2. Create DoG(Difference of Gaussian) in octaves
# Source Code Hint: buildDoGPyramid()
# the algorithm use DoG to replace LoG(Laplacian of Gaussian) to save time
# in step 1-1, images in each octave are GaussBlur result
# the algorithm need is "Difference of Gaussian"
# to make the different gauss image need subtract adjacent GaussBlur image
# in the document illustrated, each octave count from 5 to 4 after subtract
# in the source code , nOctaveLayers+3 become nOctaveLayers+2 in function buildDoGPyramidComputer
#
# 1-3. Find extreme point from multiple layer space
# Source Code Hint: findScaleSpaceExtremaComputer()-operator()
# Key Role: make the SIFT get Scale Invariable ability
# after preparing the DoG images in an octave of one scale
# search the extreme point of local
# comparing the pixel near the target point to find the extreme value
# near area means the 8 neighbor pixels, the 9 pixels above, and the 9 pixels underneath
# which means 1 pixel should compare with 8+9+9 = 26 pixels in close space
# from source code you can see a huge if condition near the line 635
# then take a threshold to filter the potential points with extrema value
#
# 2. Key point Localization
# 2-1. Refine the potential points to keypoint
# Source Code Hint: adjustLocalExtrema()
# the step 1-3 extreme points are not the keypoints
# because of edges effect we need to refine the keypoints to get more accurate points
# the method is to use Taylor series expansion (to understand this, we need more mathematic knowledge)
# then pick up a threshold to filter the result
# the result of stage1 is the keypoint in the Pyramids * Octaves space, not a flat image
#
# 2-2. Flatten the keypoints to origin size coordinate
# Source Code Hint: adjustLocalExtrema()
# the stage2 is flattening those points to the origin image coordinate
# so that we can do the "Orientation Assignment" step
# near line 569, you can examine assigning of point (x, y)
#
# 3. Orientation Assignment
# 3-1. Use 36 bins histogram to calculate the main direction
# Source Code Hint: calcOrientationHist()
# Key Role: make the SIFT get Rotate Invariable ability
# loop the keypoint, remove the duplicate point
# A neighborhood is taken around the keypoint, calculate the gradient
# divide 360 degrees to 36 bins to generate a histogram
# take the max value as the main direction
# keep the 80% value of max as an associate direction
# then whatever the image rotate
# the same area main orientation will not change
#
# 4. Keypoint Descriptor
# 4-1. Neighborhood pixels & Rotate
# Source Code Hint: calcOrientationHist()
# once the orientation is made
# use the 16*16 pixel square with the keypoint as center calculate descriptor
# divide the 16*16 square to 16 small square (call it "block") with 4*4 size
# each pixel in the block calculate the gradient
# then 16 gradient number is token at one block
#
# 4-2. The second histogram with 8 bins
# Source Code Hint: calcSIFTDescriptor() + SIFT_DESCR_HIST_BINS
# now we need a new histogram to represent the block
# not the same histogram in step 3-1
# block histogram use 8 bins represent 360 degree
# the reason using 8 bins is there are only 16 candidate gradient in a block
#
# 4-3. Assemble the block histogram to the descriptor
# Source Code Hint: calcSIFTDescriptor() + float* dst
# calculate each block histogram with 8 bins
# store the 8 bins value to last parameter dst
# 16 * 16 divided to 16 blocks
# the final descriptor is a vector with 16 * 8 = 128 size
# every 8 elements of this vector represent a block histogram
#
# 5. Keypoint Matching
# this step is not the SIFT algorithm itself
# it emphasizes algorithm is used for stitching images
# should pick an appropriate threshold to get a better result
# sometimes the closet-distance is not as good as the second candidate
#
# Refer:
# https://blog.csdn.net/zddblog/article/details/7521424
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)

# draw keypoints
keypoint = img.copy()
cv.drawKeypoints(gray, kp, keypoint)
orientation = img.copy()
cv.drawKeypoints(gray, kp, orientation, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# show
plt.subplot(2, 2, 1), plt.imshow(img), plt.title("img")
plt.subplot(2, 2, 2), plt.imshow(keypoint), plt.title("keypoint")
plt.subplot(2, 2, 3), plt.imshow(orientation), plt.title("orientation")
plt.show()
