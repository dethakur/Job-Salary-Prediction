__author__ = 'devashishthakur'

import matplotlib
from matplotlib import pyplot as plt

f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/NaiveBayes/results-MNB-3.txt')
print(f)
# for line in f:
#     print(line)

map_el = {}
for line in f:
    arr = line.split(",")
    map_el[float(arr[0])] = line

xAxis = []
trainArr = []
testArr = []
for key_val in map_el.keys():
    line = map_el[key_val]
    arr = line.split(",")
    xAxis.append(key_val)
    # trainArr.append(arr[1])
    testArr.append(1-float(arr[2]))

print(sorted(xAxis))
# plt.plot(xAxis,trainArr,'+',color='green')
plt.plot(xAxis,testArr,'+',color='blue')

plt.show()
# print(map_el.keys())