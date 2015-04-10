from sklearn.feature_extraction.text import *
import csv
import scipy.sparse as sp

import pickle

vector_features = 2000

vectorizerMap = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/vectorizer-{}.pkl'.format(vector_features),'rb'))
X = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/X-{}.pkl'.format(vector_features),'rb'))
# Y = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y-{}.pkl'.format(vector_features),'rb'))
Y  = pickle.load(open("/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y_classified-{}.pkl".format(vector_features),'rb'))
X_test = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/X-test-{}.pkl'.format(vector_features),'rb'))
Y_test  = pickle.load(open("/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y_classified-test-{}.pkl".format(vector_features),'rb'))

import numpy as np

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
    
C = 5
Y = np.array([i[0] for i in Y])
Y_test = np.array([i[0] for i in Y_test])

for i in range(0.0001,3,0.0003):
    logistic = linear_model.LogisticRegression(C=0.01,multi_class='ovr',solver='liblinear')

# from sklearn import linear_model, decomposition
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
#
# labels = ['Class'+str(i) for i in range(1,6)]
# logistic = linear_model.LogisticRegression(C=0.01,multi_class='ovr',solver='liblinear')
# logistic.fit(X,Y)
# print('Fitting done , now predicting')
# Y_pred = logistic.predict(X_test)
# print('Prediction done')
# print(Y_pred)
# print(classification_report(Y_test, Y_pred, target_names=labels))
# print(confusion_matrix(Y_test, Y_pred))