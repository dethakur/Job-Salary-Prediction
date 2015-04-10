# filePath = 'data-sets/temporary.csv'

from sklearn.feature_extraction.text import *
import csv
import scipy.sparse as sp

import pickle

vectorizerMap = {}

# Path for training data csv file
filePath = '/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/data-sets/Train_rev1.csv'

train_data = []
test_data_index = []

f_file = open(filePath)
l_val = list(csv.reader(f_file,delimiter=","))

total_data_size = len(l_val)-1
all_indices = range(1,total_data_size)
import numpy as np
# Find 60% of random data size
random_indices = list(np.random.choice(all_indices,int(0.6*total_data_size),replace=False))
remainining_indices = list(set(all_indices) - set(random_indices))
training_map = {}
for el in random_indices:
    training_map[el] = 1


map_el = {}
X = {}
Y = []

feature_arr = []
for line in l_val[0:1]:
    for i in range(0,len(line)):
        feature_arr.append(line[i])


count =1
for line in l_val[1:]:
    if not count in training_map:
        count +=1
        continue

    for i in range(0,len(line)):
        key = feature_arr[i]
        if key not in map_el:
            map_el[key] = []

        map_el[key].append(line[i])

    count +=1

from scipy.sparse import csr_matrix
salary = [int(el) for el in map_el['SalaryNormalized']]
salary = csr_matrix(salary)
salary = salary.T
Y = salary

del map_el['Id']
del map_el['SalaryRaw']
del map_el['ContractType']

array_indices = []

# Deleting the column that contains salary range and contract type. Contract Type is empty


for key in map_el.keys():
    if key != 'SalaryNormalized':
        value = map_el[key]
        # Each column needs to have its own vectorizer
        vectorizerMap[key] = TfidfVectorizer(max_df=1.0, max_features=1000,
                                 min_df=0, stop_words='english',
                                 use_idf=True)
        X[key] = vectorizerMap[key].fit_transform(value)



output_arr = []
for key in X.keys():
    output_arr.append(X[key])

X = sp.hstack(output_arr, format='csr')
print(X.shape)

pickle.dump(vectorizerMap,open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/vectorizer-1000.pkl','wb'))
pickle.dump(X,open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/X-1000.pkl','wb'))
pickle.dump(Y,open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y-1000.pkl','wb'))

pickle.dump(remainining_indices,open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/test_indices_1000.pkl','wb'))


__author__ = 'devashishthakur'

import pickle
import numpy as np

X = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/X-1000.pkl','rb'))
Y = pickle.load(open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y-1000.pkl','rb'))

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


pickle.dump(y_arr,open("/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y_classified-1000.pkl",'wb'))
pickle.dump(dataBucketMap,open("/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/regression_bucket.pkl-1000",'wb'))