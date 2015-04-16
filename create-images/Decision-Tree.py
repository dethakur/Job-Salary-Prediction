__author__ = 'devashishthakur'

import matplotlib
from matplotlib import pyplot as plt

f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/Decison-Trees/results-DT-Classifier-3.txt')
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
    trainArr.append(1-float(arr[1]))
    testArr.append(1-float(arr[2]))

print(map_el[max(xAxis)])
print(map_el[min(xAxis)])
# plt.plot(xAxis,trainArr,'+',color='green')

plt.xlabel("Height of the tree")
plt.ylabel("CV Error")

plt.plot(xAxis,trainArr,'+',color='red',label='Training Error')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.plot(xAxis,testArr,'+',color='blue',label='CV error')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)


plt.show()
# print(map_el.keys())