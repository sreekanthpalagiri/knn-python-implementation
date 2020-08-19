import numpy as np
import math
import os

dir=os.getcwd()
datadir=dir+"\\data\\regression\\"

""" 
Method the returns distance between two features
"""
def euclideandist(p,q):
    dist=0
    for a,b in zip(p,q):
        dist+=(a-b)**2
    return math.sqrt(dist)

""" 
Method the returns R2 score
"""
def calcr2(y_pred,y_test):
    y_hat=np.mean(y_test)
    ssr=(y_pred-y_test)**2
    tss=(y_hat-y_test)**2
    rsq=1-(np.sum(ssr)/np.sum(tss))
    return rsq*100
        
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
Method return predicted value using weighted distance calculation, we have used N=2 (passed to method)
Pseudo code: calculates weighted distance formula sum(1/(distance**N)*y_trainvalue)/sum(1/(distance**N))
"""

def weighdist(eucdist,srtorder,y_train,N):
    y_pred=0
    d=0
    for a, b in zip(eucdist[srtorder],y_train[srtorder]):
        y_pred += (1/(a**N))*b[0]
        d+=(1/(a**N))
    
    y_pred=y_pred/d
    #print(y_pred,y_train[srtorder],srtorder)
    return(y_pred)

""" 
Takes training and testfile names as input
"""
print('Please input training file name (full name with .csv), file should be in \\data\\regression\\')
trainingfilename=input()

print('Please input test file name (full name with .csv), file should be in \\data\\regression\\')

testfilename=input()

print('Please input value of K')
k=input()
if k=='':
    print('Treating K == 10')
    k=10

k=int(k)

if trainingfilename == '':
    trainingfilename = 'trainingData.csv'
    print('Treating training file as trainingData.csv')
if testfilename == '':
    testfilename = 'testData.csv'
    print('Treating test file as testfilename.csv')

""" 
Loads training and test data
"""
trainingdata = np.genfromtxt(datadir+trainingfilename, delimiter=',')
testdata = np.genfromtxt(datadir+testfilename, delimiter=',')

""" 
Stores feature values in x and class values in y by test and train data. Y_pred is empty numpy array used to populate prediction data
"""
x_train=trainingdata[:,0:12]
y_train=trainingdata[:,12:13]
x_test=testdata[:,0:12]
y_test=testdata[:,12:13].ravel()
y_pred=np.empty(len(y_test))

""" 
For each row in test, get dist and srt order. Calls method which does distance weighted calculation for prediction. 
We only pass k rows to the method.
"""
for index, testrow in enumerate(x_test):
    eucdist, srtorder = calculatedistances(x_train,testrow)
    y_pred[index]=weighdist(eucdist,srtorder[0:k],y_train,3)
    
""" 
Print Class Predicted vs Class from test data 50 each time for comparison
"""
print('Printing Class Predicted vs Class from test data, 50 each time for comparison:')
for i in range(round(len(y_pred)/50)):
    print('Predicted Class:', y_pred[i*50:(i+1)*50])
    print('Test Data Class:', y_test[i*50:(i+1)*50])

""" Calculating Accuracy, 1 if value in y_pred, y_test are same else 0 """
Accuracy=calcr2(y_pred,y_test)
print('Accuracy of Nearest Neighbour for K=',k,':', round(Accuracy,2),'%')