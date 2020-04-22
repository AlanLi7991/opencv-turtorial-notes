import numpy as np
import cv2 as cv
import glob


# Q1: Why do we need to pose estimation?
#   from document referring to "how the object is situated in space, like how it is rotated"
#   what means to know how an object place in 3D space, but render on 2D plane image
#   the core idea is converting a 3D point to a 2D pixel point
#   it is a project from 3D coordinate to 2D coordinate
#
# Q2: How to estimate?
# convert the problem to
#   "where the camera position in 3D space,
#   if shot the image(chessboard) vertical in Z,
#   and parallel in XY plane"
#   then from the document said
#   "we can assume Z=0, such that,
#   the problem now becomes how the camera is placed
#   in space to see our pattern image."
#
# Q3: How does it work?
#   1. we just prepare some vertex
#   2. find rotate & transform vectors from camera matrix & distort coefficients
#   3. then project vertex to pixel use vectors
#   4. draw those vertexes
#

def drawAxis(img, corners, img_pts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(img_pts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(img_pts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(img_pts[2].ravel()), (0, 0, 255), 5)
    return img


def drawCube(img, corners, img_pts):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground floor in green
    img = cv.drawContours(img, [img_pts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(img_pts[i]), tuple(img_pts[j]), (255), 3)
    # draw top layer in red color
    img = cv.drawContours(img, [img_pts[4:]], -1, (0, 0, 255), 3)
    return img


# Load previously saved data
with np.load('camera.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj_pt = np.zeros((6*7, 3), np.float32)
obj_pt[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# prepare axis vertex
#
# X axis is drawn from (0,0,0) to (3,0,0)
# so for Y axis
# For Z axis, it is drawn from (0,0,0) to (0,0,-3).
# Negative denotes it is drawn towards the camera.
axis = np.float32([
    [3, 0, 0],
    [0, 3, 0],
    [0, 0, -3]
])
axis = axis.reshape(-1, 3)

# prepare cube vertex
cube = np.float32([
    [0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
    [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]
])

# loop
for fname in glob.glob('left*.jpg'):
    # read
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # get corners
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    #
    if ret == True:
        # find float32 accuracy of corners
        subpix_corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(obj_pt, subpix_corners, mtx, dist)
        # project 3D points(axis points) to image plane(pixel coordinate)
        img_pts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # draw 3D model on the plane image
        axis_ret = img.copy()
        axis_ret = drawAxis(axis_ret, subpix_corners, img_pts)
        # same logic draw cube
        img_pts, jac = cv.projectPoints(cube, rvecs, tvecs, mtx, dist)
        cube_ret = img.copy()
        cube_ret = drawCube(cube_ret, subpix_corners, img_pts)
        # show
        cv.imshow('axis_ret', axis_ret)
        cv.imshow('cube_ret', cube_ret)
        # wait key to next
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite(fname[:6]+'.png', img)

cv.destroyAllWindows()
