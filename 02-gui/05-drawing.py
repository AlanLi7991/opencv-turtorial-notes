import numpy as np
import cv2 as cv

# Create a black image
img = np.zeros((512, 512, 3), np.uint8)

# draw line:
#   on img
#   from (0, 0)
#   to (511, 511)
#   use bgr space (255, 0 , 0) color
#   with thickness 5px
cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

# draw rectangle:
#   on img
#   (384, 0) as left-top
#   (510, 128) as right-bottom
#   use bgr space (0, 255, 0) color
#   with thickness 3px
cv.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

# draw circle:
#   on img
#   (447, 63) as center
#   63 as radius
#   use bgr space (0, 0, 255) color
#   thickness -1 means fill with color
cv.circle(img, (447, 63), 63, (0, 0, 255), -1)

# draw ellipse:
#   on img
#   (256, 256) as center
#   (100, 50) as horizontal axis, vertical axis
#   use 0° rotation angle
#   draw from 0° to 180°
#   use bgr space (255, 0, 0) color
#   thickness -1 means fill with color
cv.ellipse(img, (256, 256), (100, 50), 0, 0, 180, 255, -1)


# create vertex points
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
# reshape to ROWSx1x2
pts = pts.reshape((-1, 1, 2))
# draw polylines
#   on img
#   array of multiple polylines[pts, pts2, pts3...], we draw only pts
#   use isClose = True
#   use bgr space (0, 255, 255) color
cv.polylines(img, [pts], True, (0, 255, 255))

# define a font
font = cv.FONT_HERSHEY_SIMPLEX
# draw text
#   on img
#   text string OpenCV
#   (10, 500) as bottom-left corner
#   HERSHEY font face
#   4 times scale of base size
#   bgr space (255, 255, 255) white color
#   with thickness 2px
#   antialiased line type
cv.putText(img, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 2, cv.LINE_AA)

# show image
cv.imshow("image", img)

# opencv gui control
cv.waitKey()
cv.destroyWindow("image")
