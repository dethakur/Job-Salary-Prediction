__author__ = 'devashishthakur'

import matplotlib
from matplotlib import pyplot as plt

f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/Decison-Trees/results.txt')

xAxis = []
trainArr = []
testArr = []

map_el = {}

for line in f:
    arr = line.rstrip().split(",")
    key_val = arr[0]
    map_el[key_val] = line.rstrip()

for key in map_el.keys():
    line = map_el[key]
    arr = line.split(",")
    xAxis.append(arr[0])
    trainArr.append(arr[1])
    testArr.append(arr[2])


plt.xlabel("Height of the Decision Tree")
plt.ylabel("Regression Error")
plt.plot(xAxis,trainArr,'+',color='blue',label='Training Error',alpha=1.0)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.plot(xAxis,testArr,'+',color='black',label='5 Fold Cross Validation Error',alpha=1.0)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.plot([25,25],[0,max(trainArr)],'-',color='red')
plt.show()

# f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/Decison-Trees/results-cv-3.txt')
#
# xAxis = []
# trainArr = []
# testArr = []
#
# map_el = {}
#
# for line in f:
#     arr = line.rstrip().split(",")
#     key_val = arr[0]
#     map_el[key_val] = line.rstrip()
#
# for key in map_el.keys():
#     line = map_el[key]
#     arr = line.split(",")
#     xAxis.append(arr[0])
#     trainArr.append(arr[1])
#     testArr.append(arr[2])
#
#
# plt.plot(xAxis,trainArr,'+',color='m',label='Training Error CV 10',alpha=1.0)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#            ncol=2, mode="expand", borderaxespad=0.)
#
# plt.plot(xAxis,testArr,'+',color='k',label='Cross Validaion Error -5 Fold',alpha=1.0)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#            ncol=2, mode="expand", borderaxespad=0.)
#
#
# plt.show()
#
#
#
