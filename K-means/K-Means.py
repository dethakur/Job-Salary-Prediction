# filePath = 'data-sets/temporary.csv'

from sklearn.feature_extraction.text import *
import csv
import scipy.sparse as sp

import pickle

vectorizerMap = {}
vecture_features = 2000

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

output_arr.append(Y)
X = sp.hstack(output_arr, format='csr')
print(X.shape)
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

output_arr.append(Y_test)  
X_test = sp.hstack(output_arr, format='csr')
print(X_test.shape)

from sklearn.cluster import KMeans

print('Data has been read')

f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/K-means/results-K-Means-3.txt','a')

import time

for k in range(3,100):
    start = time.clock()
    print('Clustering starts')
    clf = KMeans(n_clusters=k)
    clf.fit(X)
    print('Time taken for cluster = {} is = {}'.format(k,time.clock()-start))
    print("Iteration = {} and cluster centers = {}".format(k,clf.cluster_centers_))
    f.write("{},{}\n".format(k,clf.inertia_))
    f.flush()


