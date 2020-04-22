import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# GMM paper:
# http://leap.ee.iisc.ac.in/sriram/teaching/MLSP_16/refs/GMM_Tutorial_Reynolds.pdf
#
# reference source code comment
# ref: http://cvml0824.is-programmer.com/posts/41253.html
#
# Q1: What is GrabCut?
# "Grab Cut" is an interactive segmentation algorithm derived from "Graph Cut"
# interact means needing user to manually brush the background & foreground
#
# Q2: What is GMM?
# "Gaussian Mixture Models", a descriptor using multiple gaussian component,
# each gaussian component has it own (means, covariance, weight)
# assemble M component result is a feature vector
#
# Q3: What store GMM must use shape (1, 65)?
# 1 pixel has 3 channels, 1 weight component
# 3 channels make it has 3 means
# each change has a covariance with each mean make it has 3*3 covariance
# modelSize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/
# 1 GMM has 5 components(OpenCV hard code, don't know why), component size is modelSize
# finnally, the size of parameters will be  5*(3+9+1) = 65
#
# Q4: How to comfirm M component?
# ordinary data has it's gaussian distribution,
# GMM use k-means classifier to get K cluster,
# for each cluster, it will get it's gaussian distribution
# then pick M (M <= N) as the feature
#
#

# read
img = cv.imread('gisele.jpg')
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# shape
shape = img.shape[:2]

# create first mask container
# this mask will be modified by cv.grabCut
mask = np.zeros(shape, np.uint8)

# crate array to store GMM Model histogram
# those arrays will be modified by cv.grabCut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# create rect wrap gisele, manually calculated
rect = (125, 20, 700, 700)

# use grab cut algorithm with GC_INIT_WITH_RECT
# mask, bgdModel, fgdModel will be modified
# afterward, mask has four values 0, 1, 2, 3 represnet cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD
# afterward, bgdModel & fgdModel will be assigned by float64 probability histogram?
cv.grabCut(rgb, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

# copy result for show , gray multiple 100
mask_rect_img = mask.copy()

# create mask non interactive
# assign cv.GC_BGD(0) cv.GC_PR_BGD(2) to 0, cv.GC_FGD(1) cv.GC_PR_FGD(3) to 1
mask_ni = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# copy result for show , gray multiple 100
mask_ni_img = mask_ni.copy()

# apply mask, get non interactive result
mask_3d = mask_ni[:, :, np.newaxis]
result_ni = rgb*mask_3d

# grab is the mask image I manually labelled
grab = cv.imread('gisele_grab.jpg', 0)

# wherever it is marked black (sure background), change mask=0
mask[grab == 0] = 0
# wherever it is marked white (sure foreground), change mask=1
mask[grab == 255] = 1

# copy result for show , gray multiple 100
mask_grab_img = mask.copy()

# use grab cut algorithm with GC_INIT_WITH_MASK type
# in this type
# the mask parameter has four values
# 0: the cv.GC_BGD calculated result, grab == 0 pixels
# 1: the cv.GC_FGD calculated result, grab == 1 pixels
# 2: the cv.GC_PR_BGD calculated result, may be substracted by grab == 0 pixels
# 3: the cv.GC_PR_FGD calculated result, may be substracted by grab == 1 pixels
#
# the rect parameter pass None, calculate base on mask value
cv.grabCut(rgb, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)

# create mask after interactively grab
mask_i = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# copy result for show , gray multiple 100
mask_i_img = mask_i.copy()

# apply mask, get interactive result
result_i = rgb*mask_i[:, :, np.newaxis]

# show
images = [
    rgb, mask_rect_img, mask_ni_img,
    result_ni, mask_grab_img, mask_i_img,
    result_i
]
colors = [
    None, "gray", "gray",
    None, "gray", "gray",
    None
]
titles = [
    "rgb", "mask after rect", "mask none",
    "result none",  "mask after grab", "mask interactive",
    "result interactive"
]
for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], colors[i])
    plt.title(titles[i])
plt.show()
