import numpy as np
import cv2 as cv

# Load the data, converters convert the letter to a number
data = np.loadtxt(
    'letter-recognition.data', dtype='float32', delimiter=',',
    converters={0: lambda ch: ord(ch)-ord('A')}
)

# split the data to two, 10000 each for train and test
train, test = np.vsplit(data, 2)

# split trainData and testData to features and responses
responses, trainData = np.hsplit(train, [1])
labels, testData = np.hsplit(test, [1])

# Initiate the kNN, classify, measure accuracy.
knn = cv.ml.KNearest_create()

# train data with responses
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)

# use testData as input 
ret, result, neighbours, dist = knn.findNearest(testData, k=5)

# count the correct number in matches shape
correct = np.count_nonzero(result == labels)
# calculate the accuracy of knn
accuracy = correct*100.0/10000

# print
print('''
1. load letter-recognition.data line by line to 20000 samples
2. first column is result alphabet, next 16 columns is feature vector
3. use 10000 as train data, 
4. use 10001-20000 as test data
5. split first column as responses
6. use test data as input
7. compare the result with test_labels prepared

the last accuracy ratio is %s
''' % accuracy)