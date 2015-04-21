import pickle
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn import cross_validation

from sklearn.cross_validation import StratifiedKFold
import numpy as np

import time

clf = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/km-cluster-clf.pkl','rb'))
dataXMap = {}
dataYMap = {}
for el in range(0,10):
    X = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/km-cluster-X{}.pkl'.format(el),'rb'))
    dataXMap[el] = X 
    Y = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/km-cluster-Y{}.pkl'.format(el),'rb'))
    dataYMap[el] = Y


# print(X[labelMap[0]].shape)

cluster_to_error_map = {}
for cluster_id in range(0,10):	

	X = dataXMap[cluster_id]
	Y = dataYMap[cluster_id]

	X = X.toarray()
	Y = Y.toarray()

	Y = np.array([i[0] for i in Y])
	
	print(X.shape)
	print(Y.shape)

	#cluster_id = 0
	cluster_to_error_map[cluster_id] = {}
	for ht in range(1,15):
	    cluster_to_error_map[cluster_id][ht] = {}
	    dlf = tree.DecisionTreeRegressor(max_depth=ht)
	    skf = cross_validation.StratifiedKFold(Y, n_folds=5,shuffle=True)
	    print(skf)
	    print()
	    test_error = []
	    train_error = []
	    # print("for iteration {}".format(ht))
	    for train_index, test_index in skf:
	       print("for iteration {}".format(ht))
	       X_train, X_test = X[train_index], X[test_index]
	       y_train, y_test = Y[train_index], Y[test_index]
	        
	       dlf.fit(X_train,y_train)
	        
	       y_pred = dlf.predict(X_test)
	       test_error.append(mean_absolute_error(y_pred,y_test))
	       
	       y_pred = dlf.predict(X_train)
	       train_error.append(mean_absolute_error(y_pred,y_train))
	    
	    
	    print('Done for cluster_id = {} and ht = {}'.format(cluster_id,ht))
	    print("{},{},{},{}".format(cluster_id,ht,sum(train_error)/len(train_error),sum(test_error)/len(test_error)))
	    cluster_to_error_map[cluster_id][ht]['train'] = sum(train_error)/len(train_error)
	    cluster_to_error_map[cluster_id][ht]['test'] = sum(test_error)/len(test_error)    
	    
for key in cluster_to_error_map.keys():
	print('Info about cluster Id = ',key)
	for ht in cluster_to_error_map[key].keys():
		train_err = cluster_to_error_map[key][ht]['train']
		test_err = cluster_to_error_map[key][ht]['test']

		print("{},{},{}".format(ht,train_err,test_err))
	
	print('-----xx-----xxx-------xxxx-------')
	     
pickle.dump(cluster_to_error_map,open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/km-DT-ClusterToErrorMap.pkl','wb'))	    
	    


