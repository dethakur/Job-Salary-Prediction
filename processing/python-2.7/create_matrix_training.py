# filePath = 'data-sets/temporary.csv'

from sklearn.feature_extraction.text import *
import csv
import scipy.sparse as sp

import pickle

vectorizerMap = {}

# Path for training data csv file
filePath = '../../data-sets/Train_rev1.csv'

f_file = open(filePath)
l_val = list(csv.reader(f_file,delimiter=","))
map_el = {}
X = {}
Y = []

feature_arr = []
for line in l_val[0:1]:
    for i in range(0,len(line)):
        feature_arr.append(line[i])
print(feature_arr)
for line in l_val[1:]:
    for i in range(0,len(line)):
        key = feature_arr[i]
        if key not in map_el:
            map_el[key] = []
        
        map_el[key].append(line[i])

# Deleting the column that contains salary range and contract type. Contract Type is empty
del map_el['Id']
del map_el['SalaryRaw']
del map_el['ContractType']

for key in map_el.keys():
    if key != 'SalaryNormalized':
        value = map_el[key]
        # Each column needs to have its own vectorizer
        vectorizerMap[key] = TfidfVectorizer(max_df=1.0, max_features=1000,
                                 min_df=0, stop_words='english',
                                 use_idf=True)
        X[key] = vectorizerMap[key].fit_transform(value)
 
from scipy.sparse import csr_matrix
salary = [int(el) for el in map_el['SalaryNormalized']]
salary = csr_matrix(salary)
salary = salary.T
Y = salary

output_arr = []
for key in X.keys():
    output_arr.append(X[key])
    
X = sp.hstack(output_arr, format='csr')
print(X.shape)

pickle.dump(vectorizerMap,open('../../pkl-files/python-2.7/vectorizer-1000.pkl','w'))
pickle.dump(X,open('../../pkl-files/python-2.7/X-1000.pkl','w'))
pickle.dump(Y,open('../../pkl-files/python-2.7/Y-1000.pkl','w'))

