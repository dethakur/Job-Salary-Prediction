

print(__doc__)

# import pickle
# X = pickle.load(open('../pkl-files/python-2.7/X.pkl'))
# Y = pickle.load(open('../pkl-files/python-2.7/Y_classified.pkl'))

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)
errorMap = {}

for i in frange(0.01,10,0.01):

    logreg = linear_model.LogisticRegression(C=i,penalty='l2')

    # we create an instance of Neighbours Classifier and fit the data.
    logreg.fit(X, Y)
    # print(X.shape)
    # print(Y.shape)
    z = logreg.predict(X)
    a = np.zeros(z.shape[0])
    a[z!=Y] =1
    error = sum(a)/a.shape[0]
    print(error*100)
    errorMap[str(i)] = error

import matplotlib
from matplotlib import pyplot as plt

x = errorMap.keys()
y = errorMap.values()
x = [float(p) for p in x]
y = [float(p) for p in y]

plt.plot(x,y,'*')
plt.show()