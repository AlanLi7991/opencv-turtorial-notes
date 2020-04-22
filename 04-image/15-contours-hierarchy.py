import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread('hierarchy_1.png', cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# threshold
ret, thresh = cv.threshold(gray, 127, 255, 0)

# find contours
enums = (cv.RETR_LIST, cv.RETR_TREE, cv.RETR_EXTERNAL, cv.RETR_CCOMP)
titles = ["cv.RETR_LIST", "cv.RETR_TREE", "cv.RETR_EXTERNAL", "cv.RETR_CCOMP"]
for idx, enum in enumerate(enums):
    ret_img, contours, hierarchy = cv.findContours(thresh, enum, cv.CHAIN_APPROX_SIMPLE)
    print("hierarchy: \n", hierarchy, "\n")
    n = len(contours)
    con_img = img.copy()
    for i in range(n):
        cv.drawContours(con_img, contours, i, (255, 0, 0), 2)
    plt.subplot(2, 2, idx+1)
    plt.imshow(con_img)
    plt.title(titles[idx])
plt.show()
