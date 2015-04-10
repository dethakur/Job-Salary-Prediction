__author__ = 'devashishthakur'

import pickle
from sklearn.feature_extraction.text import *
import csv
import scipy.sparse as sp

vectorizerMap = open('../pkl-files/python-3.0/vectorizer.pkl','rb')
vectorizerMap = pickle.load(vectorizerMap)
print(vectorizerMap.keys())

X = pickle.load(open('../pkl-files/python-3.0/X.pkl','rb'))
y = pickle.load(open('../pkl-files/python-3.0/X.pkl','rb'))

