
import numpy as np
import cv2 as cv
import matplotlib.pyplot as pyplot

# read image
img = cv.imread("lena.tif", cv.IMREAD_GRAYSCALE)

# show image
cv.imshow("image", img)

# opencv gui control
cv.waitKey()
cv.destroyWindow("image")

# matplot gui control
pyplot.imshow(img, cmap="Reds")
pyplot.show()
