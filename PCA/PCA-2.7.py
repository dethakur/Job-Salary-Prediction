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