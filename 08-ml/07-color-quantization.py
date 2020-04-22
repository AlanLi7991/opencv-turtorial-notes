import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# read
img = cv.imread('home.jpg', cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# convert all pixels to list
Z = img.copy().reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = (2, 4, 8)
for idx, k in enumerate(K):
    ret, label, center = cv.kmeans(Z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    # fill pixels with label
    pixel = label.flatten()
    # convet label to (r,g,b) tuple
    res = center[pixel]
    # reshape to image
    result = res.reshape(img.shape)

    # add quantization image 
    plt.subplot(2, 2, idx + 2)
    plt.imshow(result)
    plt.title("k = %s" % k)

# add original
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("original")

# show
plt.show()
