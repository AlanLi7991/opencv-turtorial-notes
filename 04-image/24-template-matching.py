import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# read
img_raw = cv.imread('rose_img.png', cv.IMREAD_COLOR)
img_gray = cv.cvtColor(img_raw, cv.COLOR_BGR2GRAY)
template = cv.imread('rose_red.png', cv.IMREAD_GRAYSCALE)

# get template size
width, height = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

for idx, meth in enumerate(methods):

    # convert to cv constant
    method = eval(meth)

    # apply template matching
    res = cv.matchTemplate(img_gray, template, method)

    # get location result
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)

    # stroke rect angle
    img = img_raw.copy()
    cv.rectangle(img, top_left, bottom_right, 255, 2)

    # add result
    plt.subplot(4, 3, idx+1)
    plt.imshow(res, "gray")
    plt.title(meth)

    plt.subplot(4, 3, idx+7)
    plt.imshow(img)
    plt.title(meth)


# show
plt.show()
