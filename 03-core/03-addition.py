
import numpy as np
import cv2 as cv

# read image
back = cv.imread("lena.tif", cv.IMREAD_UNCHANGED)
front = cv.imread("nico.png", cv.IMREAD_COLOR)

# clip
length = len(back)
front = front[:length, :length]
print("back size ", back.shape, " front size ", front.shape)

# add
addition = cv.add(back, front)
cv.imshow("addition", addition)
cv.waitKey()

# add weight
blending = cv.addWeighted(back, 0.7, front, 0.3, 0)
cv.imshow("blending", blending)
cv.waitKey()

# destroy
cv.destroyAllWindows()
