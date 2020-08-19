import numpy as np
import math
import os

dir=os.getcwd()
datadir=dir+"\\data\\classification\\"

""" 
Method the returns distance between two features
"""
def euclideandist(p,q):
    dist=0
    for a,b in zip(p,q):
        dist+=(a-b)**2
    return math.sqrt(dist)

""" 
Method returns max occuring class amount k neighbours
"""
def vote(y_train,sortorder):
    unique, counts = np.unique(y_train[sortorder], return_counts=True)
    return unique[np.argmax(counts)]

"""
For a single query point and a collection of data points in feature space returns:
    1) euclidean distances between a  in eucdit
    2) indexes of eucdist in ascending order of its values in srtdindex
"""
def calculatedistances(x_train,xtest):
    eucdist=np.empty(len(x_train))
    #srtdindex=np.empty((len(x_train),1))
    for index, row in enumerate(x_train):
        eucdist[index]=euclideandist(row,xtest)
    srtdindex = np.argsort(eucdist)
    return eucdist, srtdindex

""" 
Takes training and testfile names as input
"""
print('Please input training file name (full name with .csv), file should be in \\data\\classification\\')
trainingfilename=input()
print('Please input test file name (full name with .csv), file should be in \\data\\classification\\')
testfilename=input()

if trainingfilename == '':
    trainingfilename = 'trainingData.csv'
    print('Treating training file as trainingData.csv')
if testfilename == '':
    testfilename = 'testData.csv'
    print('Treating test file as testfilename.csv')

print('Please input value of K')
k=input()
if k=='':
    print('Treating K == 1')
    k=1

k=int(k)

""" 
Loads training and test data
"""
trainingdata = np.genfromtxt(datadir+trainingfilename, delimiter=',')
testdata = np.genfromtxt(datadir+testfilename, delimiter=',')

""" 
Stores feature values in x and class values in y by test and train data. Y_pred is empty numpy array used to populate prediction data
"""
x_train=trainingdata[:,0:10]
y_train=trainingdata[:,10:11]
x_test=testdata[:,0:10]
y_test=testdata[:,10:11].ravel()
y_pred=np.empty(len(y_test))

""" 
For each row in test, get dist and srt order. Class of training data at index value in srtorder[0] is the predicted class
"""
for index, testrow in enumerate(x_test):
    eucdist, srtorder = calculatedistances(x_train,testrow)
    #y_pred[index]=y_train[srtorder[0]]
    y_pred[index]=vote(y_train,srtorder[0:k])

""" 
Print Class Predicted vs Class from test data 50 each time for comparison
"""
print('Printing Class Predicted vs Class from test data, 50 each time for comparison:')
for i in range(round(len(y_pred)/50)):
    print('Predicted Class:', y_pred[i*50:(i+1)*50])
    print('Test Data Class:', y_test[i*50:(i+1)*50])


""" Calculating Accuracy, 1 if value in y_pred, y_test are same else 0 """
Accuracy=(sum(1 for x,y in zip(y_pred,y_test) if x == y) / len(y_test))*100
print('Accuracy of Nearest Neighbour for k=',k,':',round(Accuracy,2),'%')