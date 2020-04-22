import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


# read
img = cv.imread('gradients.jpg', cv.IMREAD_GRAYSCALE)

# sobel kernel
sobel_x_8 = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=5)
sobel_y_8 = cv.Sobel(img, cv.CV_8U, 0, 1, ksize=5)
sobel_x_64 = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobel_y_64 = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

# laplacian kernel use sobel inside
laplacian_8 = cv.Laplacian(img, cv.CV_8U)
laplacian_64 = cv.Laplacian(img, cv.CV_64F)


# box
box = cv.imread('box.jpg', cv.IMREAD_GRAYSCALE)

# sobel kernel convolution
sobel_x8u = cv.Sobel(box, cv.CV_8U, 1, 0, ksize=5)
sobel_x64f = cv.Sobel(box, cv.CV_64F, 1, 0, ksize=5)

# absolute float
abs_sobel64f = np.absolute(sobel_x64f)

# convert float to uint8
sobel_8u = np.uint8(abs_sobel64f)

# show
images = [
    img, sobel_x_8, sobel_y_8, 
    laplacian_8, sobel_x_64, sobel_y_64, 
    laplacian_64, box, sobel_x8u, 
    sobel_x64f, abs_sobel64f, sobel_8u
]
colors = [
    "gray", "gray", "gray", 
    "gray", "gray", "gray", 
    "gray", "gray", "gray", 
    "gray", "gray", "gray", 
    "gray", "gray", "gray"
]
titles = [
    "img", "sobel_x_8", "sobel_y_8", 
    "laplacian_8", "sobel_x_64", "sobel_y_64", 
    "laplacian_64", "box", "sobel_x8u", 
    "sobel_x64f", "abs_sobel64f", "sobel_8u"
]
for i in range(len(images)):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
plt.show()
