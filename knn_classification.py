#!/usr/bin/env python

#Classification via K-Nearest Neighbors - Supervised Learning 

#Add arguments for k

import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from sklearn import svm, datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()

x = iris.data[:, :2] #extracting the first two features
y = iris.target

colors = ['red', 'blue', 'green']
for color, i, target in zip(colors, [0,1,2], iris.target_names):
    plt.scatter(x[y==i, 0], x[y==i, 1], color=color, label=target)

plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

#Training a KNN model on Iris data 
from sklearn.neighbors import KNeighborsClassifier

#k=99 for smoother classification boundaries
k = 1 #Adjust k to avoid overfitting or underfitting

knn = KNeighborsClassifier(n_neighbors=k) #initialize model

knn.fit(x, y) #Fit the model

#Min and max for first feature 
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1

#Min and max for second feature 
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

h = (x_max / x_min) / 100 #step size in the mesh - simply used in making a range/list of points using np.arange

#Making predictions for each of the points 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

z = z.reshape(xx.shape) #Match the z shape to that of xx

#Generate the color plot 
plt.contourf(xx, yy, z, cmap=plt.cm.Accent, alpha=0.8) #Color plot without the points

#Superimpose the training points 
colors = ['red', 'blue', 'green']
for color, i, target in zip(colors, [0,1,2], iris.target_names):
    plt.scatter(x[y==i, 0], x[y==i, 1], color=color, label=target)

plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.title(f'KNN k = {k}') #Quick use of f string
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

predictions = knn.predict(x)

#Prediction and counts 
print(np.unique(predictions, return_counts=True)) 
print(iris.target_names)


