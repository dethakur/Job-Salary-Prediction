import pickle
import numpy as np
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
rf = open('/Users/devashishthakur/Documents/Machine Learning/Job-Salary-Prediction/Job-Salary-Prediction/info-files/title-w-distribution.txt','r')

count_arr = []
for line in rf:
	val = int(line.split("@@@")[1].rstrip())
	if val == 1:
		continue
	count_arr.append(val)





# example data
# mu = 100 # mean of distribution
# sigma = 15 # standard deviation of distribution

x = np.array(count_arr)
print(x.shape)
mu = np.mean(x)
sigma = np.std(x)
max_val = np.max(x)
min_val = np.min(x)
num_bins = 9
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=0, facecolor='blue', alpha=1)
# add a 'best fit' line

y = mlab.normpdf(bins, mu, sigma)
plt.title("Title Word Frequency across clusters")
plt.plot(bins, y, 'r--')
# plt.xlabel('Salary ^ 0.1')
# plt.ylabel('Count')
# plt.title(r'Histogram of Title: $\mu={}$, $\sigma={}$'.format(mu,sigma))

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()




