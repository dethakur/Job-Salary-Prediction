__author__ = 'devashishthakur'

import pickle
import numpy as np
Y = pickle.load(open("../pkl-files/python-2.7/Y.pkl"))
# print(Y)

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


# example data
# mu = 100 # mean of distribution
# sigma = 15 # standard deviation of distribution

x = Y.toarray()
# x = (np.log(x)+1)
# x = np.power(np.log(x),1)
x = np.power(x,0.1)
# print(type(x))
mu = np.mean(x)
sigma = np.std(x)
max_val = np.max(x)
min_val = np.min(x)
# print(mu)
# print(sigma)
# print(x.shape)
num_bins = 90
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Salary ^ 0.1')
plt.ylabel('Count')
plt.title(r'Histogram of Salary: $\mu={}$, $\sigma={}$'.format(mu,sigma))

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()




