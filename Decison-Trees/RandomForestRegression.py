from sklearn.feature_extraction.text import *
import csv
import scipy.sparse as sp
import numpy as np

import pickle

print('Testing')

vector_features = 2000

X = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/X-{}.pkl'.format(vector_features),'rb'))
Y = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y-{}.pkl'.format(vector_features),'rb'))
X_test = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/X-test-{}.pkl'.format(vector_features),'rb'))
Y_test  = pickle.load(open("/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y-test-{}.pkl".format(vector_features),'rb'))

Y = Y.toarray()
Y_test = Y_test.toarray()
Y = np.array([i[0] for i in Y])
Y_test = np.array([i[0] for i in Y_test])

f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/Decison-Trees/results-RandomForest-5.txt','a')
# f.write("-------Cross validation results 10 fold --------\n")
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor

from sklearn.cross_validation import StratifiedKFold

import time
for depth in range(1,25):
  prev_train_error = -1
  prev_test_error = -1
  for i in range(3,500,2):
      start = time.clock()
      clf = RandomForestRegressor(max_depth=depth,n_estimators=i)
      
      skf = cross_validation.StratifiedKFold(Y, n_folds=5,shuffle=True)
      len(skf)

      print(skf)  

      test_error = []
      train_error = []
      

      for train_index, test_index in skf:
         print("for iteration {} for depth = {}".format(i,depth))
         X_train, X_test = X[train_index], X[test_index]
         y_train, y_test = Y[train_index], Y[test_index]
         start = time.clock()

         clf = clf.fit(X_train,y_train)
         
         print('Time taken = ',(time.clock()-start))

         y_pred = clf.predict(X_test)
         test_error.append(mean_absolute_error(y_pred,y_test))
         
         y_pred = clf.predict(X_train)
         train_error.append(mean_absolute_error(y_pred,y_train))
     
      test_error = sum(test_error)/len(test_error)
      train_error = sum(train_error)/len(train_error)
      f.write('{},{},{},{}\n'.format(depth,i,train_error,test_error))
      f.flush()
      # if train_error > prev_train_error and not(prev_train_error == -1):
      #   break;
      prev_train_error = train_error

      
