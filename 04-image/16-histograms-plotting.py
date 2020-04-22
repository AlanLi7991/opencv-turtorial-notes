import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread('lena.tif', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# ravel
ravel = img.ravel()
print("ravel : \n %s \n" % (ravel))

# histogram
hist_cv = cv.calcHist([gray], [0], None, [256], [0, 256])
hist_np, bins = np.histogram(ravel, 256, [0, 256])
print("histogram : \n cv: \n %s \n np: \n %s \n" % (hist_cv, hist_np))

# show
for i, col in enumerate(('r', 'g', 'b')):
    histr = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.subplot(2, 2, i+2)
    plt.hist(histr, 256, [0, 256], color=col)
plt.subplot(2, 2, 1)
plt.hist(ravel, 256, [0, 256])
plt.show()
