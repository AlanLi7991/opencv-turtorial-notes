import numpy as np
import cv2 as cv

img_u: np.ndarray = cv.imread("lena.tif", cv.IMREAD_UNCHANGED)
img_c = cv.imread("lena.tif", cv.IMREAD_COLOR)

# read pixel 100*100
pixel_u = img_u[100, 100]
pixel_c = img_c[100, 100]
print("pixel unchanged %s, pixel color %s" % (pixel_u, pixel_c))

# print image structure
print(img_u.shape)

# show original
cv.imshow("image", img_u)
cv.waitKey()

# change pixel 100*100 to white
img_u[:100, :100] = (255, 255, 255)
cv.imshow("white", img_u)
cv.waitKey()

# Move right bottom to left top
corner = img_u[412:, 412:]
img_u[:100, :100] = corner
cv.imshow("corner", img_u)
cv.waitKey()

# Get channel
b, g, r = cv.split(img_u)
img_m = cv.merge((r, g, b))
cv.imshow("merge", img_m)
cv.waitKey()

# Close
cv.destroyAllWindows()
