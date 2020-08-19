import numpy as np
import math
import os
import sys
 

dir=os.getcwd()
datadir=dir+"\\data\\classification\\"

distancemetric = {1:'Euclidean',
            2:'Manhattan',
            3:'Minkowski(P=3)',
            4:'Chebyshev'}

""" 
Method the returns Euclidean distance between two features
"""
def Euclidean(p,q):
    dist=0
    for a,b in zip(p,q):
        dist+=(a-b)**2
    return math.sqrt(dist)

""" 
Method the returns Manhattan distance between two features
"""
def Manhattan(p,q):
    dist=0
    for a,b in zip(p,q):
        dist+=abs(a-b)
    return dist

""" 
Method the returns Minkowski distance between two features
"""
def Minkowski(p,q,N):
    dist=0
    for a,b in zip(p,q):
        dist+=abs(a-b)**N
    return math.pow(dist, 1/N)

""" 
Method the returns Minkowski distance between two features
"""
def chebyshev(p,q):
    dist=abs(p-q)
    #cos_sim = np.dot(p,q.T)/(np.linalg.norm(p)*np.linalg.norm(q))
    return np.max(dist)

"""
For a single query point and a collection of data points in feature space returns:
    1) euclidean distances between a  in eucdit
    2) indexes of eucdist in ascending order of its values in srtdindex
"""
def calculatedistances(x_train,xtest,distmet):
    dist=np.empty(len(x_train))
    for index, row in enumerate(x_train):
        if distmet==1:
            dist[index]=Euclidean(row,xtest)
        elif distmet==2:
            dist[index]=Manhattan(row,xtest)
        elif distmet==3:
            dist[index]=Minkowski(row,xtest,3)
        elif distmet==4:
            dist[index]=chebyshev(row,xtest)
    srtdindex = np.argsort(dist)
    return dist, srtdindex

"""
Method return predicted value using weighted distance calculation, we have used N=1
Pseudo code: Creates dictionary with each unique class in y_train as key and distance initial 0 as value, 
we get only first K rows of eucdist and srtorder as input into this method
calculates inverse distances against each class and stores it as value
return class with max distance value
"""

def weighdist(eucdist,srtorder,y_train,N):
    classdict={x:0 for x in np.unique(y_train[srtorder])}
    for a, b in zip(eucdist[srtorder],y_train[srtorder]):
        classdict[b[0]] = classdict[b[0]] + (1/(a**N))
    return( max(classdict, key=classdict.get))

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
    print('Treating K == 10')
    k=10
k=int(k)

print('Please select distance metric from below:')
print(distancemetric)
distmet=int(input())

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
For each row in test, get dist and srt order. Calls method which does distance weighted calculation for prediction. 
We only pass k rows to the method.
"""
for index, testrow in enumerate(x_test):
    eucdist, srtorder = calculatedistances(x_train,testrow,distmet)
    y_pred[index]=weighdist(eucdist,srtorder[0:k],y_train,1)
 
""" 
Print Class Predicted vs Class from test data 50 each time for comparison
"""
print('Printing Class Predicted vs Class from test data, 50 each time for comparison:')
for i in range(round(len(y_pred)/50)):
    print('Predicted Class:', y_pred[i*50:(i+1)*50])
    print('Test Data Class:', y_test[i*50:(i+1)*50])


""" Calculating Accuracy, 1 if value in y_pred, y_test are same else 0 """
Accuracy=(sum(1 for x,y in zip(y_pred,y_test) if x == y) / len(y_test))*100
print('Accuracy of Nearest Neighbour for K=',k,':', round(Accuracy,2),'%')