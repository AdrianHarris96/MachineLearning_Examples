#!/usr/bin/env python

#Basics of sklearn

#Generating linear data in sklearn 
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
x, y = make_regression(n_samples=100, n_features=1, noise=11) 
plt.scatter(x, y)
plt.show()

#Generating clustered data 
import numpy as np
from sklearn.datasets import make_blobs
x, y = make_blobs(1000, centers=4) #Num of points, number of clusters
rgby = np.array(['r', 'g', 'b', 'y']) #Coloring for the clusters
plt.scatter(x[:, 0], x[:, 1] , color=rgby[y])
plt.show()

#Generating circular clusters
from sklearn.datasets import make_circles
x, y = make_circles(n_samples=500, noise=0.06)
rgb = np.array(['r', 'g'])
plt.scatter(x[:, 0], x[:, 1], color=rgb[y])
plt.show()