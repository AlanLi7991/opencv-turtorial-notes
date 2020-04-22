import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread('lena.tif')

# scale 2
img2x = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

# scale 0.5
height, width = img.shape[:2]
img05x = cv.resize(img, (int(width/2), int(height/2)), interpolation=cv.INTER_CUBIC)

# translate
M_trans = np.float32([[1, 0, 100], [0, 1, 50]])
img_trans = cv.warpAffine(img, M_trans, (width, height))

# rotate
center = ((width-1)/2, (height-1)/2)
M_rotate = cv.getRotationMatrix2D(center, 90, 1)
img_rotate = cv.warpAffine(img, M_rotate, (width, height))

# affine transformation
pts1 = np.float32([
    [50, 50], [200, 50], [50, 200]
])
pts2 = np.float32([
    [10, 100], [200, 50], [100, 250]
])

M_affine = cv.getAffineTransform(pts1, pts2)
img_affine = cv.warpAffine(img, M_affine, (width, height))

# perspective transformation
pts1 = np.float32([
    [56, 65], [368, 52], [28, 387], [389, 390]
])
pts2 = np.float32([
    [0, 0], [300, 0], [0, 300], [300, 300]
])
M_per = cv.getPerspectiveTransform(pts1, pts2)
img_per = cv.warpPerspective(img, M_per, (width, height))

# show
images = [img, img2x, img05x, img_trans, img_rotate, img_affine, img_per]
for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i])
plt.show()
