import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread("blox.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
gray_float = np.float32(gray)

# harris to get potential points
dst = cv.cornerHarris(gray_float, 2, 3, 0.04)
dst = cv.dilate(dst, None)

# create mask with threshold, assign potential points to 255
ret, mask = cv.threshold(dst, 0.01*dst.max(), 255, cv.THRESH_BINARY)
mask = np.uint8(mask)

# result of harris
harris = img.copy()
harris[mask == 255] = [255, 0, 0]

# find centroids of mask
ret, labels, stats, centroids = cv.connectedComponentsWithStats(mask)

# assign the [:, 1] y axis means row number, [:, 0] x axis means column number
centroid = img.copy()
fancy = np.int0(centroids)
centroid[fancy[:, 1], fancy[:, 0]] = [0, 255, 0]

# define the criteria to stop and refine the corners
# criteria is a struct(C++) tuple(Python)
# first parameter is type, define use iteration count/ epsilon which as end
# second parameter is iteration count
# third parameter is epsilon value
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)


# refine the sub pixel corner with criteria
# use (5, 5) and (-1, -1) create a kernel to do convolution
# convolution refine the potential corners, which passed is the centroids
corners = cv.cornerSubPix(gray_float, np.float32(centroids), (5, 5), (-1, -1), criteria)

# assign the [:, 1] y axis means row number, [:, 0] x axis means column number
refine = img.copy()
fancy = np.int0(corners)
refine[fancy[:, 1], fancy[:, 0]] = [0, 0, 255]

# show
plt.subplot(2, 2, 1), plt.imshow(img), plt.title("img")
plt.subplot(2, 2, 2), plt.imshow(harris), plt.title("harris")
plt.subplot(2, 2, 3), plt.imshow(centroid), plt.title("centroid")
plt.subplot(2, 2, 4), plt.imshow(refine), plt.title("refine")
plt.show()
