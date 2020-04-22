import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# read
img = cv.imread('lena.tif', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# show original image
plt.subplot(2, 2, 3)
plt.imshow(img)
plt.title("original")

hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
# cumulative sum
cdf = hist.cumsum()
# cumulative normalized
cdf_normalized = cdf * float(hist.max()) / cdf.max()
print("histogram : \n%s\n%s\n" % (hist, cdf))

# first
plt.subplot(2, 2, 1)
plt.plot(cdf_normalized, color='b')  # line
plt.hist(img.flatten(), 256, [0, 256], color='r')  # histogram
plt.xlim([0, 256])  # x axis limit
plt.legend(('cdf', 'histogram'), loc='upper left')  # brand

# mask all zero
cdf_m = np.ma.masked_equal(cdf, 0)

# normalize to 0-255 use cdf
# not like cdf_normalized
cdf_m = ((cdf_m - cdf_m.min()) / (cdf_m.max()-cdf_m.min())) * 255

# convet to uint8
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

# use fancy-indexing create new image
# https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html
img_equal = cdf[img]

# show new image
plt.subplot(2, 2, 4)
plt.imshow(img_equal)
plt.title("img_equal")

# histogram new image gray
gray_equal = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
hist, bins = np.histogram(gray_equal.flatten(), 256, [0, 256])

# cumulative normalized
cdf_equal = hist.cumsum()
cdf_normalized_equal = cdf * float(hist.max()) / cdf.max()

# show new image
plt.subplot(2, 2, 2)
plt.plot(cdf_normalized_equal, color='b')
plt.hist(img_equal.flatten(), 256, [0, 256], color='r')
plt.legend(('cdf_normalized_equal', 'histogram'), loc='upper left')  # brand
plt.show()
