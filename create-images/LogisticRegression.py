__author__ = 'devashishthakur'

import matplotlib
from matplotlib import pyplot as plt

f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/LogisticRegression/output-value/LR-info-2000-90%.txt')
# print(f)
# for line in f:
#     print(line)

map_el = {}
for line in f:
    arr = line.split(",")
    map_el[float(arr[0])] = line

xAxis = []
trainArr = []
testArr = []
map_val = {}
for key_val in map_el.keys():
    line = map_el[key_val]
    arr = line.split(",")
    xAxis.append(key_val)
    # trainArr.append(arr[1])
    testArr.append(float(arr[1]))

print(map_el[max(xAxis)])
print(map_el[min(xAxis)])
# plt.plot(xAxis,trainArr,'+',color='green')

plt.xlabel("Value of Lambda")
plt.ylabel("CV Error")

plt.plot(xAxis,testArr,'+',color='blue')


plt.show()
# print(map_el.keys())