import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# read
img = cv.imread('lena.tif', cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# pyramid down
down = cv.pyrDown(img)
down = cv.pyrDown(down)
down = cv.pyrDown(down)

# pyramid up
up = cv.pyrUp(down)
up = cv.pyrUp(up)
up = cv.pyrUp(up)

# read apple & orange
apple = cv.cvtColor(cv.imread("apple.jpg", cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
orange = cv.cvtColor(
    cv.imread("orange.jpg", cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)

# concatenate directly
width = apple.shape[1]//2
joint = np.hstack((apple[:, :width], orange[:, width:]))

# generate Gaussian Pyramid
a = apple.copy()
o = orange.copy()
gaussian_apple = [a]
gaussian_orange = [o]
for i in range(6):
    a = cv.pyrDown(a)
    o = cv.pyrDown(o)
    gaussian_apple.append(a)
    gaussian_orange.append(o)

# sort from low resolution to high
gaussian_apple.reverse()
gaussian_orange.reverse()

# generate Laplacian Pyramid
laplacian_apple = [gaussian_apple[0]]
laplacian_orange = [gaussian_orange[0]]
for i in range(6):
    # pyramid up lost information
    a = cv.pyrUp(gaussian_apple[i])
    o = cv.pyrUp(gaussian_orange[i])
    # use standard information subtract lost information in same level
    a = cv.subtract(gaussian_apple[i+1], a)
    o = cv.subtract(gaussian_orange[i+1], o)
    # append
    laplacian_apple.append(a)
    laplacian_orange.append(o)

# loop laplacian and concatenate
concatenates = []
for a, o in zip(laplacian_apple, laplacian_orange):
    rows, cols, dpt = a.shape
    # concatenate
    con = np.hstack((a[:, 0:cols//2], o[:, cols//2:]))
    concatenates.append(con)

# now reconstruct from low to high
result = concatenates[0]
for i in range(6):
    result = cv.pyrUp(result)
    result = cv.add(result, concatenates[i+1])

# show
images = [
    img, down, up, apple, orange, joint,
    # gaussian_apple[6], gaussian_apple[5], gaussian_apple[4],
    # gaussian_orange[6], gaussian_orange[5], gaussian_orange[4],
    # laplacian_apple[6], laplacian_apple[5], laplacian_apple[4],
    # laplacian_orange[6], laplacian_orange[5], laplacian_orange[4],
    # concatenates[6], concatenates[5], concatenates[4],
    result
]
titles = [
    "img", "down", "up", "apple", "orange", "joint",
    # "gaussian_apple_256", "gaussian_apple_128", "gaussian_apple_64",
    # "gaussian_orange_256", "gaussian_orange_128", "gaussian_orange_64",
    # "laplacian_apple_256", "laplacian_apple_128", "laplacian_apple_64",
    # "laplacian_orange_256", "laplacian_orange_128", "laplacian_orange_64",
    # "concatenates_256", "concatenates_128", "concatenates_64",
    "result"
]
for i in range(len(images)):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
plt.show()
