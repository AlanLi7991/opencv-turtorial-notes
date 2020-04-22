import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.cvtColor(cv.imread('lena.tif'), cv.COLOR_BGR2RGB)

# 2d convolution
# the basic filter/blur math theory
# about convolution: https://www.zhihu.com/question/22298352
kernel = np.ones((5, 5), np.float32)/25
img_con = cv.filter2D(img, -1, kernel)

# averaging blur, simple filter
img_aver = cv.blur(img, (5, 5))

# gaussian blur, weight filter, more father more less weight
img_gau = cv.GaussianBlur(img, (5, 5), 0)

# median blur, sort filter, use odd range and pick median value
img_med = cv.medianBlur(img, 5)

# bilateral filter, very slow filter, main effect is keep edges sharp
img_bil = cv.bilateralFilter(img, 9, 75, 75)

# show
images = [img, img_con, img_aver, img_gau, img_med, img_bil]
titles = ["img", "convolution", "averaging", "gaussian", "median", "bilateral"]
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
plt.show()
