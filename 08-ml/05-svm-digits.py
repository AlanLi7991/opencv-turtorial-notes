#!/usr/bin/env python
import cv2 as cv
import numpy as np


# skew:
#   neither parallel nor at right angles to a specified or implied line; askew; crooked
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    # mu** means central moments
    # f(xy)xy / f(xy)y^2
    skew = m['mu11']/m['mu02']
    # size of image
    size = img.shape[:2]
    # what's this matrix?
    M = np.float32([
        [1, skew, -0.5*size[0]*skew],
        [0, 1, 0]
    ])
    # apply transform with
    img = cv.warpAffine(img, M, size, flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
    # return de-skew result
    return img

# hog:
#   Histogram of Oriented Gradients
def hog(img):
    # create gradient of x,y
    # shape is (20, 20)
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    # convert to polar coordinate system
    mag, ang = cv.cartToPolar(gx, gy)
    # normalize angles
    ang = ang/(2*np.pi)
    # define bins count
    bin_count = 16  
    # locate bin number with normalize result
    # quantizing binvalues in (0...16)
    bin_num = np.int32(bin_n*ang)   
    # split (20, 20) to four (10, 10) sub region
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    # distrbute magnitude & angle to bin_count histogram
    hists = [np.bincount(b.ravel(), m.ravel(), bin_count) for b, m in zip(bin_cells, mag_cells)]
    # stack 4 hists to on 64 length
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist


# read
img = cv.imread('digits.png', cv.IMREAD_GRAYSCALE)
if img is None:
    raise Exception("we need the digits.png image from samples/data here !")

# split to 20*20 pixels cells
cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

# First half is train_data, remaining is test_data
train_cells = [i[:50] for i in cells]
test_cells = [i[50:] for i in cells]

# deskew all train cell, then take result calculate hog
train_de_skewed = [list(map(deskew, row)) for row in train_cells]
test_de_skewed = [list(map(deskew, row)) for row in test_cells]

# calculate hog of de-skewed data, take hog as train data
train_hog = [list(map(hog, row)) for row in train_de_skewed]
test_hog = [list(map(hog, row)) for row in test_de_skewed]

# convert hog to float32 accuracy, then reshape to list
# with 64 dimension feature vector as item
train_data = np.float32(train_hog).reshape(-1, 64)
test_data = np.float32(test_hog).reshape(-1, 16*4)

# create responses 0~9, and repeat to 2500
responses = np.repeat(np.arange(10), 250)[:, np.newaxis]

# create svm & configuration
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

# train
svm.train(train_data, cv.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')

# label test data
ret, result = svm.predict(test_data)

# match result with correct answer
matches = result == responses
# count the correct number in matches shape
correct = np.count_nonzero(matches)
# calculate the accuracy of knn
accuracy = correct*100.0/result.size

# print
print('''
1. split digits.png to 5000 cells
2. use 2500 as train cells
3. use 2501-5000 as test cells
4. process all cell with de-skewed & create histogram
5. take the result of histogram as train & test data
6. create 0-9 labels as responses
7. train svm with train data
8. use test data as input
9. compare the result with correct answer well already known

the last accuracy ratio is %s
''' % accuracy)
