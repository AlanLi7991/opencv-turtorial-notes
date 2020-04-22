import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# read
img = cv.imread('lena.tif')

# like this: aaaaaa | abcdefgh | hhhhhhh
replicate = cv.copyMakeBorder(img, 20, 20, 20, 20, cv.BORDER_REPLICATE)
# like this: fedcba | abcdefgh | hgfedcb
reflect = cv.copyMakeBorder(img, 20, 20, 20, 20, cv.BORDER_REFLECT)
# like this: gfedcb | abcdefgh | gfedcba
reflect101 = cv.copyMakeBorder(img, 20, 20, 20, 20, cv.BORDER_REFLECT_101)
# like this: cdefgh | abcdefgh | abcdefg
wrap = cv.copyMakeBorder(img, 20, 20, 20, 20, cv.BORDER_WRAP)
# constant value
constant = cv.copyMakeBorder(img, 20, 20, 20, 20, cv.BORDER_CONSTANT, value=(255, 255, 255))

# show
plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

plt.show()
