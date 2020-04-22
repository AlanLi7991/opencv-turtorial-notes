import numpy as np
import cv2 as cv
import glob
import os.path

# Q1: Why need to calibrate?
#   Because photo will be distorted after camera shot, the reason is light refract
#
# Q2: What kinds of distortion exist?
#   Two major kinds of distortion are RADIAL DISTORTION(径向畸变) and TANGENTIAL DISTORTION(切向畸变).
#
# Q3: Which distortion will effect image
#   1. Radial distortion causes straight lines to appear curved.
#   2. Tangential distortion causes "some areas in the image may look nearer than expected."
#
# Q4: What caused distortion?
# The actual reason is pinhole cameras theory design, the physical reason is light refraction
#   1. radial distortion occurs because light has different lengths to pinhole via different refraction
#   2. tangential distortion occurs because the image-taking lens is not aligned perfectly parallel to the imaging plane.
#
# Q5: What is needed for correct distortion?
#   Need intrinsic and extrinsic parameters
#   Intrinsic parameters are specific to a camera.
#   Extrinsic parameters correspond to rotation and translation vectors
#   which translates a coordinates of a 3D point to a coordinate system.
#
# Q6: Intrinsic parameters contain?
#   They include information like focal length (fx,fy) and optical centers (cx,cy).
#
#
#

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# 1. create 42 points with value (0, 0, 0)
obj_pt = np.zeros((6*7, 3), np.float32)
# 2. create nparray with shape (2, 7, 6)
point_order = np.mgrid[0:7, 0:6]
# 3. transform the matrix to shape (6, 7, 2)
#    it means create 6 * 7 matrix with (x, y) points as item
point_order = point_order.T
# 4. reshape to (42, 2), it means one dimension list of points 
#    the value of point is (0, 0) (1, 0) ... (5, 5) (6, 5)
point_order = point_order.reshape(-1, 2)
# 5. assign the 3 dimension points x, y value of (x, y, z) 
#    although the points are 3D,
#    they all lie in the calibration pattern's XY coordinate plane
#    (thus 0 in the Z-coordinate)
#    only need assign x, y
obj_pt[:, :2] = point_order

# Arrays to store object points and image points from all the images.
obj_pts = []  # 3d point in real world space
img_pts = []  # 2d points in image plane.

# read image lists 
images = glob.glob('left*.jpg')

# loop
for image in images:
    # read image & convert gray
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find the chess board corners
    # same size with obj_pt
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        # add obj_pt of this index image
        # all images use same obj_pt to find correlation 
        obj_pts.append(obj_pt)
        # find float32 accuracy of corners
        subpix_corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # add point pixel value of this index image
        img_pts.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7, 6), subpix_corners, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

# close all
cv.destroyAllWindows()


# calibrate camera 
# ret: return value
# matrix:
# camera matrix combine with fx, fy, cx, cy and zero
# dist
# Input/output vector of distortion coefficients 
# (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements.
# rvecs:
# Output vector of rotation vectors (Rodrigues) estimated for each pattern view
# tvecs:
# Output vector of translation vectors estimated for each pattern view
size = gray.shape[::-1]
ret, matrix, dist, rvecs, tvecs = cv.calibrateCamera(obj_pts, img_pts, size, None, None)
if os.path.exists("camera.npz") == False:
    np.savez("camera.npz", mtx=matrix, dist=dist, rvecs=rvecs, tvecs=tvecs)

# prepare a image without correct distort
img = cv.imread('left12.jpg')
h, w = img.shape[:2]

# use camera matrix & distortion coefficients refine the matrix
# alpha:
# if the scaling parameter alpha=0, it may even remove some pixels at image corners
# if alpha=1, all pixels are retained with some extra black images.
correct_matrix, roi = cv.getOptimalNewCameraMatrix(matrix, dist, (w, h), 1, (w, h))

# There are two methods to correct
# 1. Using cv.undistort()
# 2. Using remapping
#
# Camera matrix of the distorted image.  
# correct distort
# Q: Why need both matrix & correct_matrix ?
# by default, correct_matrix it is the same as camera matrix
# but you may additionally scale and shift the result by using a different matrix.
# it is what we done on getOptimalNewCameraMatrix()
dst = cv.undistort(img, matrix, dist, None, correct_matrix)

# crop the image
# Q: Why need crop process?
# after undistort image some pixels will be lost
# and generate none pixels boards, normally present with black color
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# show 
cv.imshow('undistort result', dst)
cv.waitKey()

# using remapping correct distort
# create map with camera matrix & distortion coefficients
(mapx, mapy) = cv.initUndistortRectifyMap(matrix, dist, None, correct_matrix, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# un distort with rectify map
cv.imshow('undistort rectify map result', dst)
cv.waitKey()

# close all
cv.destroyAllWindows()
