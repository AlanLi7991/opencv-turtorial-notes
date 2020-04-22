import numpy as np
import cv2 as cv


def blurTime(optimize: bool):
    # read
    img = cv.imread("lena.tif", cv.IMREAD_UNCHANGED)

    # set optimized
    cv.setUseOptimized(optimize)

    # mark tick
    e1 = cv.getTickCount()

    # median filter must use odd ksize
    for i in range(1, 100, 2):
        img = cv.medianBlur(img, i)
        pass

    # mark tick
    e2 = cv.getTickCount()

    # deduction
    t = (e2 - e1)/cv.getTickFrequency()

    # print
    print("use optimized ", optimize, " time ", t)

    # show
    cv.imshow("medianBlur", img)
    cv.waitKey()
    cv.destroyAllWindows()


# do
blurTime(0)
blurTime(1)
