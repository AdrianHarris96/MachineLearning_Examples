#!/usr/bin/env python

#SVM Classification for 3D space 

#Plotting 3D Hyperplane 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles

#Generate features and class labels
feats, c_labels = make_circles(n_samples=500, noise=0.09)
z = feats[:, 0] ** 2 + feats[:, 1] ** 2

rgb = np.array(['r', 'g'])

fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(feats[:, 0], feats[:, 1], z, color=rgb[c_labels])
#plt.show()

#Combine feats and z into ndarray
features = np.concatenate((feats, z.reshape(-1,1)), axis=1)

from sklearn import svm

clf = svm.SVC(kernel = 'linear')
clf.fit(features, c_labels)

x3 = lambda x,y: (-clf.intercept_[0] - clf.coef_[0][0] * clf.coef_[0][1] * y) / clf.coef_[0][2]

tmp = np.linspace(-1.5, 1.5, 100)
ax.view_init(40, 20) #Change viewing angle with this snippet
#Great information on 3D plotting here -> https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

x,y = np.meshgrid(tmp, tmp)
ax.plot_wireframe(x, y, x3(x,y), color='black')
plt.show()