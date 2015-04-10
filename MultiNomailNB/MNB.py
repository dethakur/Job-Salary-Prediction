from sklearn.feature_extraction.text import *
import csv
import scipy.sparse as sp

from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import StratifiedKFold
import time

import pickle

print('Testing')

vector_features = 2000

vectorizerMap = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/vectorizer-{}.pkl'.format(vector_features),'rb'))
X = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/X-{}.pkl'.format(vector_features),'rb'))
# Y = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y-{}.pkl'.format(vector_features),'rb'))
Y  = pickle.load(open("/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y_classified-{}.pkl".format(vector_features),'rb'))
X_test = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/X-test-{}.pkl'.format(vector_features),'rb'))
Y_test  = pickle.load(open("/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y_classified-test-{}.pkl".format(vector_features),'rb'))

print(X_test.shape)
print(Y_test.shape)
import numpy as np

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
    
C = 5
Y = np.array([i[0] for i in Y])
Y_test = np.array([i[0] for i in Y_test])

labels = ['class '+str(i) for i in range(1,6)]

from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score


from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/NaiveBayes/results-MNB-3.txt','a')
import time
y = Y
for i in frange(1,45,0.1):
    
    skf = cross_validation.StratifiedKFold(y, n_folds=3,shuffle=True)
    len(skf)
    start = time.clock()
    print(skf)  
    clf = MultinomialNB(alpha=i)
    
    test_error = []
    train_error = []
    for train_index, test_index in skf:
       print("for iteration {}".format(i))
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = Y[train_index], Y[test_index]
       clf = clf.fit(X_train,y_train)
        
       y_pred = clf.predict(X_test)
       test_error.append(accuracy_score(y_pred,y_test))
       # print('Train information = ')
       # print(confusion_matrix(y_pred,y_test)) 

       # print("----------------------------------------")

       y_pred = clf.predict(X_train)
       train_error.append(accuracy_score(y_pred,y_train))
       # print('Train information = ')
       # print(confusion_matrix(y_pred,y_train)) 

        
    


    print('Time to fit the dataset of alpha = {} is {}'.format(i,time.clock()-start))
#     y_pred = clf.predict(X)
#     train_error = mean_absolute_error(y_pred,Y)

#     y_pred = clf.predict(X_test)
    test_error = sum(test_error)/len(test_error)
    train_error = sum(train_error)/len(train_error)
    f.write('{},{},{}\n'.format(i,train_error,test_error))
    f.flush()