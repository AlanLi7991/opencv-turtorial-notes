import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# This is a functional inpaint API offered by OpenCV
#
# Q1: What is inpaint
# repair the old photos
#
# Q2: popular algorithm of inpaint
# cv.INPAINT_TELEA:
#   **"An Image Inpainting Technique Based on the Fast Marching Method"** by Alexandru Telea in 2004
# cv.INPAINT_NS:
#    **"Navier-Stokes, Fluid Dynamics, and Image and Video Inpainting"** by Bertalmio, Marcelo, Andrea L. Bertozzi, and Guillermo Sapiro in 2001
#
# Q3: the keypoint of two algorithms
# cv.INPAINT_TELEA:
#   fill the average color of neighborhood
# cv.INPAINT_NS:
#   find boundary first, then fill color with neighbor
#
#

# read
img = cv.imread('gisele.jpg', cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# degraded img read
degraded = cv.imread('gisele_degraded.jpg', cv.IMREAD_COLOR)
degraded = cv.cvtColor(degraded, cv.COLOR_BGR2RGB)

# read mask for fix
mask = cv.imread("gisele_mask.jpg", cv.IMREAD_GRAYSCALE)

# inpaint
result = cv.inpaint(degraded, mask, 3, cv.INPAINT_TELEA)

# show
images = [img, degraded, mask, result]
colors = [None, None, "gray", None]
titles = ["img", "degraded", "mask", "result"]
for i in range(len(images)):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], colors[i])
    plt.title(titles[i])
plt.show()
