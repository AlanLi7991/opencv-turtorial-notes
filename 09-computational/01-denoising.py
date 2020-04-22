import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# This is a refine API offered by OpenCV
#
# "In earlier chapters, we have seen many image smoothing techniques
#  like Gaussian Blurring, Median Blurring etc and
#  they were good to some extent in removing small quantities of noise."
#
# cv.fastNlMeansDenoising() - works with a single grayscale images
# cv.fastNlMeansDenoisingColored() - works with a color image.
# cv.fastNlMeansDenoisingMulti() - works with image sequence captured in short period of time(grayscale images)
# cv.fastNlMeansDenoisingColoredMulti() - same as above, but for color images.
#
# important parameters
# h:
#   parameter deciding filter strength. Higher h value removes noise better,
#   but removes details of image also. (10 is ok)
# hForColorComponents:
#   same as h, but for color images only. (normally same as h)
# templateWindowSize:
#   should be odd. (recommended 7)
# searchWindowSize:
#   should be odd. (recommended 21)

# read image
img = cv.imread('nico.png', cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# denosie with colored API
ret_img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# read video
cap = cv.VideoCapture('vtest.avi')

# create a list of first 5 frames
frame = [cap.read()[1] for i in range(5)]

# convert all to grayscale
gray = [cv.cvtColor(i, cv.COLOR_BGR2GRAY) for i in frame]

# convert all to float64
gray = [np.float64(i) for i in gray]

# create a noise of variance 25
noise = np.random.randn(*gray[1].shape)*10

# Add this noise to images
noisy = [i+noise for i in gray]

# Convert back to uint8
noisy = [np.uint8(np.clip(i, 0, 255)) for i in noisy]

# Denoise 3rd frame considering all the 5 frames
ret_frame = cv.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)

# show
images = [img, ret_img, gray[2], noisy[2], ret_frame]
colors = ['gray', 'gray', 'gray', 'gray', 'gray']
titles = ["img", "ret_img", "gray", "noisy", "ret_frame"]
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], colors[i])
    plt.title(titles[i])
plt.show()