import cv2 as cv
import argparse

# Q1: What is Cascade Classifier?
# from the document
# "It is
#   a machine learning based approach
#   where a cascade function trained
#   from a lot of positive and negative images
#   then used to detect objects in other images"
#
# in simple word, it finds objects on the input image by trained data
#
# Q2: What cascade explanation? What is the explanation of the ...?
# from dictionary explain
# "a small waterfall, typically one of several that fall in stages down a steep rocky slope"
# the keypoint is "stages" just as same as in the algorithm logic
#
#
# Q3: Why the algorithm will generate beyond 160000 features?
# the features calculate base on kernel matrix
# not just like the contours detector only use one kernel, maybe the sobel
#       (one type) * (one size)
# in the algorithm it uses
#       (differ type) * (differ size)
# then the result will bigger than 160000 at 24x24 window size
#
#
# Q4: Why the algorithm use 6000 features as last?
# even tough algorithm get 160000 features, they are not equally useful for result
# some of them are total nonsense, we discard them,
# select the features with a minimum error rate and assign a weight component
#
# correct ratio = w1*feature1 + w2*feature2 + ..... + w6000*feature6000
#
# the paper says
#   "Even 200 features provide detection with 95% accuracy.
#    Their final setup had around 6000 features."
#
# Q5: What's the main problem of the algorithm?
# the main problem of the algorithm is too much feature and computer
#   1. we need different sizes and types kernel for convolution
#   2. each correct ratio involves too much feature needs to be evaluated one by one
# the two problems cost too much time, we need speed up.
#
# Q6: How to speed up too many features?
# use the integral image, combine the individual image in one by fill 0 of empty pixels
# like the issue:
# https://datasciencechalktalk.com/2019/07/16/haar-cascade-integral-image/
#
#
# Q6: How to speed up to correct predict?
# that why the algorithm call "CASCADE"
# it separates the feature in different groups,
# if the first group failed, it fails all
#
# document refer:
#   "The authors' detector had 6000+ features
#    with 38 stages with 1, 10, 25, 25 and 50 features
#    in the first five stages."
#
# Q8: How can I get my own *.xml trained file?
# OpenCV provides 4 program:
#
#   * opencv_createsamples
#   * opencv_annotation
#   * opencv_traincascade
#   * opencv_visualisation.
#
# for training the *.xml file， and give a tutorial in the next chapter
#
#

# prepare trained data file
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

# create cascade classifier
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

# -- 1. Load the cascades with prepared trained file model
if not face_cascade.load(args.face_cascade):
    print('--(!)Error loading face cascade')
    exit(0)

if not eyes_cascade.load(args.eyes_cascade):
    print('--(!)Error loading eyes cascade')
    exit(0)

# get default camera
camera_device = args.camera

# -- 2. Read the video stream
cap = cv.VideoCapture(camera_device)

# check device opened
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

# while
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    # -- Detect faces
    faces = face_cascade.detectMultiScale(gray)

    # loop faces
    for (x, y, w, h) in faces:
        # draw face
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        # get eyes roi
        faceROI = gray[y:y+h, x:x+w]

        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)

        # loop eyes
        for (x2, y2, w2, h2) in eyes:
            # draw eyes
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
    # show result
    cv.imshow('Capture - Face detection', frame)

    # wait 30ms
    key = cv.waitKey(30)

    # break loop with esc
    if key == 27:
        break
