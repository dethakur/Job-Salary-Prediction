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

C = 5
Y = np.array([i[0] for i in Y])
Y_test = np.array([i[0] for i in Y_test])

labels = ['class '+str(i) for i in range(1,6)]

from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/NaiveBayes/1500-results-MNB-3.txt','w')
import time
y = Y

import time
start = time.clock()
from sklearn.grid_search import GridSearchCV
x = []
for i in range(1,10):
	x.append(i)

tuned_parameters = [{'alpha':x}]


clf = GridSearchCV(MultinomialNB(), tuned_parameters, cv=3 , scoring='f1_weighted')
clf.fit(X,Y)
y_pred = clf.predict(X)
print(clf.best_params_)
print(classification_report(y_pred,Y))
print(confusion_matrix(y_pred,Y))
print(accuracy_score(y_pred,Y))

y_pred = clf.predict(X_test)
print(classification_report(y_pred,Y_test))
print(confusion_matrix(y_pred,Y_test))

	
 #    print("Detailed classification report:")
 #    print()
 #    print("The model is trained on the full development set.")
 #    print("The scores are computed on the full evaluation set.")
 #    print("---------------")
print('Time taken = {}'.format(time.clock()-start))



