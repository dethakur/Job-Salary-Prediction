__author__ = 'devashishthakur'

f = open('/Users/devashishthakur/Documents/Machine Learning/ML-Project/ML-Code/K-means/results-K-Means-3.txt');
# print(f)
import math;
from matplotlib import pyplot as plt
yAxis = []
xAxis = []
for line in f:
    arr = line.rstrip().split(",");
    xAxis.append(arr[0])
    yAxis.append(float(arr[1]))


plt.xlabel("Number of Clusters")
plt.ylabel("Cluster Inertia")
plt.plot(xAxis,yAxis,color='blue',linewidth=2,linestyle='--');
plt.plot([10,10],[max(yAxis),min(yAxis)],'-',color='red')
plt.show()



