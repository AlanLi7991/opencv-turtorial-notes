import numpy as np
import cv2 as cv

# read image
back = cv.imread("lena.tif", cv.IMREAD_UNCHANGED)
front = cv.imread("nico.png", cv.IMREAD_COLOR)

# clip
length = len(back)
front = front[:length, :length]
print("back size ", back.shape, " front size ", front.shape)

# convert front gray
img_to_gray = cv.cvtColor(front, cv.COLOR_BGR2GRAY)
cv.imshow("img_to_gray", img_to_gray)
cv.waitKey()

# make mask
ret, mask = cv.threshold(img_to_gray, 10, 255, cv.THRESH_BINARY)
cv.imshow("mask", mask)
cv.waitKey()

# make invert mask
mask_inv = cv.bitwise_not(mask)
cv.imshow("mask_inv", mask_inv)
cv.waitKey()

# pick mask on front
front_cut = cv.bitwise_and(front, front, mask=mask)
cv.imshow("front_cut", front_cut)
cv.waitKey()

# pick invert mask on back
back_cut = cv.bitwise_and(back, back, mask=mask_inv)
cv.imshow("back_cut", back_cut)
cv.waitKey()

# front + back
result = cv.add(back_cut, front_cut)
cv.imshow("result", result)
cv.waitKey()

# destroy
cv.destroyAllWindows()
