import numpy as np
import cv2 as cv

# read train data
img = cv.imread('digits.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each 20x20 size
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]

# Make it into a Numpy array. It size will be (50,100,20,20)
x = np.array(cells)
# Now we prepare train_data and test_data.
train = x[:, :50].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)  # Size = (2500,400)

# Create labels(responses) for train and test data
k = np.arange(10)
# repeat 0-9 at 250 times, then create same shape as labels(responses)
train_labels = np.repeat(k, 250)[:, np.newaxis]
# prepare the test_data labels(responses) for calculate accuracy
test_labels = train_labels.copy()

# Initiate kNN, ,
knn = cv.ml.KNearest_create()

# train the data
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)

# then test it with test data for k=1
ret, result, neighbours, dist = knn.findNearest(test, k=5)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result == test_labels
# count the correct number in matches shape
correct = np.count_nonzero(matches)
# calculate the accuracy of knn
accuracy = correct*100.0/result.size

# print
print('''
1. split digits.png to 5000 cells
2. use 2500 as train data
3. use 2501-5000 as test data
4. create 0-9 labels as responses
5. use test data as input
6. compare the result with test_labels prepared

the last accuracy ratio is %s
''' % accuracy)
