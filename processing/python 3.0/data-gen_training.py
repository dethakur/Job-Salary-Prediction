# filePath = 'data-sets/temporary.csv'

from sklearn.feature_extraction.text import *
import csv
import scipy.sparse as sp

import pickle

vectorizerMap = {}
vecture_features = 4000

# Path for training data csv file
filePath = '/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/data-sets/Train_rev1.csv'

train_data = []
test_data_index = []

f_file = open(filePath)
l_val = list(csv.reader(f_file,delimiter=","))

total_data_size = len(l_val)-1
all_indices = range(1,total_data_size)
import numpy as np
# Find 80% of random data size
random_indices = list(np.random.choice(all_indices,int(0.8*total_data_size),replace=False))
remainining_indices = list(set(all_indices) - set(random_indices))
training_map = {}
for el in random_indices:
    training_map[el] = 1

    
    


map_el = {}
test_map_el = {}
X = {}
X_test = {}
Y = []
Y_test = []

feature_arr = []
for line in l_val[0:1]:
    for i in range(0,len(line)):
        feature_arr.append(line[i])


count =1
for line in l_val[1:]:
    if not count in training_map:
        count +=1
        
        for i in range(0,len(line)):
            key = feature_arr[i]
            if key not in test_map_el:
                test_map_el[key] = []

            test_map_el[key].append(line[i])
        
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

from scipy.sparse import csr_matrix
salary = [int(el) for el in test_map_el['SalaryNormalized']]
salary = csr_matrix(salary)
salary = salary.T
Y_test = salary


del map_el['Id']
del map_el['SalaryRaw']
del map_el['ContractType']
del map_el['SalaryNormalized']

array_indices = []

# Deleting the column that contains salary range and contract type. Contract Type is empty


for key in map_el.keys():
    if key != 'SalaryNormalized':
        n = vecture_features
        if key == 'FullDescription':
            n = 10000
        value = map_el[key]
        # Each column needs to have its own vectorizer
        vectorizerMap[key] = TfidfVectorizer(max_features=n,
                                stop_words='english',
                                 use_idf=True)
        X[key] = vectorizerMap[key].fit_transform(value)



output_arr = []
for key in X.keys():
    output_arr.append(X[key])

X = sp.hstack(output_arr, format='csr')
print(X.shape)
print(Y.shape)
print("---")

del test_map_el['Id']
del test_map_el['SalaryRaw']
del test_map_el['ContractType']
del test_map_el['SalaryNormalized']

array_indices = []


#Now doing for the test data

for key in test_map_el.keys():
    if key != 'SalaryNormalized':
        value = test_map_el[key]
        # Each column needs to have its own vectorizer
        X_test[key] = vectorizerMap[key].transform(value)



output_arr = []
for key in X_test.keys():
    output_arr.append(X_test[key])

X_test = sp.hstack(output_arr, format='csr')
print(X_test.shape)
print(Y_test.shape)



pickle.dump(vectorizerMap,open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/vectorizer-{}.pkl'.format(vecture_features),'wb'))
pickle.dump(X,open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/X-{}.pkl'.format(vecture_features),'wb'))
pickle.dump(Y,open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y-{}.pkl'.format(vecture_features),'wb'))
pickle.dump(X_test,open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/X-test-{}.pkl'.format(vecture_features),'wb'))
pickle.dump(Y_test,open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y-test-{}.pkl'.format(vecture_features),'wb'))

# pickle.dump(remainining_indices,open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/test_indices_{}.pkl'.format(vecture_features),'wb'))

y_arr = Y.toarray()
y_arr = np.power(y_arr,0.1)
y_test_arr = Y_test.toarray()
y_test_arr = np.power(y_test_arr,0.1)

test_arr = np.zeros(y_test_arr.shape)
train_arr = np.zeros(y_arr.shape)
mean_val = np.mean(y_arr)
std_val = np.std(y_arr)

length = train_arr.shape[0]
test_length = test_arr.shape[0]
dataBucketMap = {}
for i in range(1,6):
    dataBucketMap[i] = []

for i in range(0,length):
    if(y_arr[i] > mean_val - 0.5*std_val and y_arr[i] <mean_val + 0.5*std_val):
        train_arr[i] = 1
        dataBucketMap[1].append(X[i])
    elif (y_arr[i] > mean_val - 1*std_val and y_arr[i] < mean_val + 1*std_val):
        train_arr[i] = 2
        dataBucketMap[2].append(X[i])
    elif (y_arr[i] > mean_val - 2*std_val and y_arr[i] < mean_val + 2*std_val):
        train_arr[i] = 3
        dataBucketMap[3].append(X[i])
#     elif (y_arr[i] > mean_val - 3*std_val and y_arr[i] < mean_val + 3*std_val):
#         train_arr[i] = 4
#         dataBucketMap[4].append(X[i])
    else:
        train_arr[i] = 4
        dataBucketMap[4].append(X[i])

        
for i in range(0,test_length):
    if(y_test_arr[i] > mean_val - 0.5*std_val and y_test_arr[i] <mean_val + 0.5*std_val):
        test_arr[i] = 1

    elif (y_test_arr[i] > mean_val - 1*std_val and y_test_arr[i] < mean_val + 1*std_val):
        test_arr[i] = 2

    elif (y_test_arr[i] > mean_val - 2*std_val and y_test_arr[i] < mean_val + 2*std_val):
        test_arr[i] = 3

#     elif (y_test_arr[i] > mean_val - 3*std_val and y_test_arr[i] < mean_val + 3*std_val):
#         test_arr[i] = 4

    else:
        test_arr[i] = 4

        
pickle.dump(test_arr,open("/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y_classified-test-{}.pkl".format(vecture_features),'wb'))
pickle.dump(train_arr,open("/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/Y_classified-{}.pkl".format(vecture_features),'wb'))
pickle.dump(dataBucketMap,open("/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/pkl-files/python-3.0/regression_bucket-{}.pkl".format(vecture_features),'wb'))