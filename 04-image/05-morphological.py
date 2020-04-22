import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# read
img = cv.imread('morphological.png', cv.IMREAD_GRAYSCALE)

# kernel make 5 x 5 x 1 shape matrix use 1 as element
kernel = np.ones((5, 5), np.uint8)

# erosion, set 0 if kernel not all positive
erosion = cv.erode(img, kernel, iterations=1)

# dilation, set 1 if kernel has at least one positive
dilation = cv.dilate(img, kernel, iterations=1)

# opening, 𝚍𝚜𝚝=open(𝚜𝚛𝚌,𝚎𝚕𝚎𝚖𝚎𝚗𝚝)=dilate(erode(𝚜𝚛𝚌,𝚎𝚕𝚎𝚖𝚎𝚗𝚝))
img_open = cv.imread("opening.png", cv.IMREAD_GRAYSCALE)
opening = cv.morphologyEx(img_open, cv.MORPH_OPEN, kernel)

# closing, 𝚍𝚜𝚝=close(𝚜𝚛𝚌,𝚎𝚕𝚎𝚖𝚎𝚗𝚝)=erode(dilate(𝚜𝚛𝚌,𝚎𝚕𝚎𝚖𝚎𝚗𝚝))
img_close = cv.imread("closing.png", cv.IMREAD_GRAYSCALE)
closing = cv.morphologyEx(img_close, cv.MORPH_CLOSE, kernel)

# gradient, 𝚍𝚜𝚝=morph_grad(𝚜𝚛𝚌,𝚎𝚕𝚎𝚖𝚎𝚗𝚝)=dilate(𝚜𝚛𝚌,𝚎𝚕𝚎𝚖𝚎𝚗𝚝)−erode(𝚜𝚛𝚌,𝚎𝚕𝚎𝚖𝚎𝚗𝚝)
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

# tophat, 𝚍𝚜𝚝=tophat(𝚜𝚛𝚌,𝚎𝚕𝚎𝚖𝚎𝚗𝚝)=𝚜𝚛𝚌−open(𝚜𝚛𝚌,𝚎𝚕𝚎𝚖𝚎𝚗𝚝)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

# blackhat, 𝚍𝚜𝚝=blackhat(𝚜𝚛𝚌,𝚎𝚕𝚎𝚖𝚎𝚗𝚝)=close(𝚜𝚛𝚌,𝚎𝚕𝚎𝚖𝚎𝚗𝚝)−𝚜𝚛𝚌
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

# create elliptical/circular shaped kernels by cv2
morph_rect = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
morph_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
morph_cross = cv.getStructuringElement(cv.MORPH_CROSS, (5, 5))
print(morph_rect, "\n", morph_ellipse, "\n", morph_cross)

# show
images = [img, erosion, dilation, img_open, opening, img_close, closing, gradient, tophat, blackhat]
titles = ["img", "erosion", "dilation", "img_open", "opening", "img_close", "closing", "gradient", "tophat", "blackhat"]
for i in range(len(images)):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i], "gray")
    plt.title(titles[i])
plt.show()
