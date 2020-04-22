import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#
#
# Q1: What is WaterShed?
# the algorithm from an idea
# "any grayscale image can be viewed as a topographic surface
# where high intensity denotes peaks and hills
# while low intensity denotes valleys."
#
# Wiki: Inter-pixel watershed[edit]
# S. Beucher and F. Meyer introduced an algorithmic inter-pixel implementation of the watershed method, [5] given the following procedure:
# 1. Label each minimum with a distinct label. Initializing a set of S with the labeled nodes.
# 2. Extract from S a node x of minimal altitude F, that is to say F(x) = min{F(y) | y ∈ S}. Attribute the label of x to each non-labeled node y adjacent to x, and insert y in S.
# 3. Repeat Step 2 until S is empty.
#
# Q2: How does the watershed algorithm work?
# 1. use dilate to find a certain background area
# 2. use distance transform to find a certain foreground area
# 3. label unknown area as low altitude (label = 0)
# 4. label certain area as high altitude (label = 1...n)
# 5. flood the water from a label, turn unknown area(0) to certain (1..n)
# 6. repeat the step5 from the lowest altitude pixel position,
# 7. step6 makes sure the waterflood always start from valleys
# 8. after all unknown area to get an altitude label
# 9. extract the high altitude as foreground, low altitude as background
#
# Q2: How to control water flood?
# as the Wiki describe
#
# with the label, the algorithm will not begin from a random position
#
# the 1 ... n label denotes a start position to flood
# the 0 label denotes the points which are not labeled y in step2
#
# when img_color labeled a position and inject it to S
# then it is not the minimum altitude of the next
#
# it will start from a position labeled, not unknown, not random
# repeat and repeat, it will work like the waterflood merging to the contours
#
# finally the contours is watershed, opencv assign it to -1
#
#
#
#
# read
img = cv.imread("coins.jpg", cv.IMREAD_COLOR)
img_color = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# binary
ret, img_otsu = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# remove noise
kernel = np.ones((5, 5), np.float32)/25
img_smooth = cv.filter2D(img_otsu, -1, kernel)

# erode & dilate
img_erode = cv.erode(img_smooth, kernel)
img_dilate = cv.dilate(img_smooth, kernel)

# direct transform find sure foreground
img_direct = cv.distanceTransform(img_erode, cv.DIST_L2, 5)
min_value = img_direct.max() * 0.7
max_value = img_direct.max()
ret, img_core = cv.threshold(img_direct, min_value, max_value, cv.THRESH_BINARY)

# substract a bounds area, which not sure for back/front
img_core = np.uint8(img_core)
img_bounds = cv.subtract(img_dilate, img_core)

# calculate a marker
ret, img_labled = cv.connectedComponents(img_core)
# label all area from 1 to ...., the background is 1
img_marker = img_labled + 1
# then assign the img_bounds to 0, which means unknown area is 0, with out label
img_marker[img_bounds == 255] = 0

# create a marker copy() for show, watershed will change marker
marker = img_marker.copy()
# apply watershed
img_watershed = cv.watershed(img_color, marker)

# prepare a result image
img_result = img_color.copy()
# watershed algorithm will assign -1 to contours, then stroke color
img_result[img_marker == -1] = [255, 0, 0]

# show
images = [
    img_color, img_gray, img_otsu, img_smooth,
    img_erode, img_dilate, img_direct,
    img_core, img_bounds,
    img_labled, img_marker,
    img_watershed, img_result
]
colors = [
    None, "gray", "gray", "gray",
    "gray", "gray", "gray",
    "gray", "gray",
    "gray", "jet",
    "gray", None
]
titles = [
    "img_color", "img_gray", "img_otsu", "img_smooth",
    "img_erode", "img_dilate", "img_direct",
    "img_core", "img_bounds",
    "img_labled", "img_marker",
    "img_watershed", "img_result"
]
for i in range(len(images)):
    plt.subplot(3, 5, i+1)
    plt.imshow(images[i], colors[i])
    plt.title(titles[i])
plt.show()
