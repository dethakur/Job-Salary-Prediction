__author__ = 'devashishthakur'

import pickle
import numpy as np

X = pickle.load(open('../../pkl-files/python-2.7/X-1000.pkl','rb'))
Y = pickle.load(open('../../pkl-files/python-2.7/Y-1000.pkl','rb'))

y_arr = Y.toarray()
y_arr = np.power(y_arr,0.1)

test_arr = np.zeros(y_arr.shape)
mean_val = np.mean(y_arr)
std_val = np.std(y_arr)

length = test_arr.shape[0]
dataBucketMap = {}
for i in range(1,6):
    dataBucketMap[i] = []

for i in range(0,length):
    if(y_arr[i] > mean_val - 0.5*std_val and y_arr[i] <mean_val + 0.5*std_val):
        test_arr[i] = 1
        dataBucketMap[1].append(X[i])
    elif (y_arr[i] > mean_val - 1*std_val and y_arr[i] < mean_val + 1*std_val):
        test_arr[i] = 2
        dataBucketMap[2].append(X[i])
    elif (y_arr[i] > mean_val - 2*std_val and y_arr[i] < mean_val + 2*std_val):
        test_arr[i] = 3
        dataBucketMap[3].append(X[i])
    elif (y_arr[i] > mean_val - 3*std_val and y_arr[i] < mean_val + 3*std_val):
        test_arr[i] = 4
        dataBucketMap[4].append(X[i])
    else:
        test_arr[i] = 5
        dataBucketMap[5].append(X[i])


pickle.dump(y_arr,open("../../pkl-files/python-2.7/Y_classified-1000.pkl",'w'))
pickle.dump(dataBucketMap,open("../../pkl-files/python-2.7/regression_bucket.pkl-1000",'w'))