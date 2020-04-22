import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# output space name
flags = [i for i in dir(cv) if i.startswith('COLOR_')]
print(flags)

# read image
img = cv.imread("lena.tif", cv.IMREAD_COLOR)

# convert hsv
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# threshold pick red hue
#
# Tips: How to find HSV values to track? 
# green = np.uint8([[[0, 255, 0]]])
# hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
#
lower = np.array([5, 100, 100])
upper = np.array([10, 255, 255])

# create mask
mask = cv.inRange(hsv, lower, upper)

# mask img
res = cv.bitwise_and(img, img, mask=mask)

# show
plt.subplot(221)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('img')

plt.subplot(222)
plt.imshow(hsv)
plt.title('hsv')

plt.subplot(223)
plt.imshow(mask, 'gray')
plt.title('mask')

plt.subplot(224)
plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
plt.title('res')

plt.show()
