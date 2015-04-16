__author__ = 'devashishthakur'

import matplotlib
from matplotlib import pyplot as plt

f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/Adaboost/results-MNB-5.txt')
print(f)
# for line in f:
#     print(line)

map_el = {}
for line in f:
    arr = line.split(",")
    map_el[int(arr[0])] = line

xAxis = []
trainArr = []
testArr = []
map_val = {}
print(map_el.keys())

for key_val in map_el.keys():
    line = map_el[key_val]
    arr = line.split(",")

    xAxis.append(key_val)
    trainArr.append(1-float(arr[1]))

    testArr.append(1-float(arr[2]))

print(sorted(xAxis))
# plt.plot(xAxis,trainArr,'+',color='green')
plt.xlabel("Number Of Iterations")
plt.ylabel("Errors")
plt.plot(xAxis,trainArr,color='blue',label='Train Error')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.plot(xAxis,testArr,color='red',label='Test Error')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.show()
# print(map_el.keys())