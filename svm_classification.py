#!/usr/bin/env python

#Classification with Support Vector Machines 

import pandas as pd
import numpy as np 
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import argparse as ap

parser = ap.ArgumentParser()
parser = ap.ArgumentParser(prog = 'Classification through SVM', description='Exploring the use of different kernels to classify points of the iris dataset')
parser.add_argument("-l", "--linear", help="invoke flag for use of the linear kernel", action='store_true', required=False)
parser.add_argument("-r", "--rbf", help="invoke flag for use of the rbf kernel", action='store_true', required=False)
parser.add_argument("-p", "--poly", help="invoke flag for use of the poly kernel", action='store_true', required=False)
args = parser.parse_args()

#ToDo: Incorporate flags for C (penalty parameter) and degree (for polynomial kernel)

iris = datasets.load_iris()

print(iris.feature_names)
print(iris.target_names)

#Extract first 2 features 
x = iris.data[:, 0:2]
y = iris.target

#Plotting scatter plot 
colors = ['red', 'green', 'blue']
for color, i, target in zip(colors, [0,1,2], iris.target_names):
    plt.scatter(x[y==i, 0], x[y==i, 1], color=color, label=target) #scatter plot via a loop - neat!
    #break

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc='best', shadow='False', scatterpoints=1)
plt.title('sepal width vs. sepal length')
plt.show()

if args.linear:
	clf = svm.SVC(kernel='linear', C=1).fit(x,y) #penalty parameter of the error term 
	#High C - fixated on classifying points correctly - low margin - 10**10
	#Low C - fixate on widest margin - likelihood of some incorrect classification - 10**-10

	#Min and max for the first feature 
	x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1

	#Min and max for the second feature 
	y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

	#step size in the mesh 
	h = (x_max / x_min)/100

	#Make predictions for each of the points 
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	#print(xx)
	z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) #Remember, ravel simply converts to a rank 1 array 

	z = z.reshape(xx.shape)
	#Plotting using contourf function
	plt.contourf(xx, yy, z, cmap=plt.cm.Accent, alpha=0.5)

	#Plot training points 
	colors = ['red', 'green', 'blue']
	for color, i, target in zip(colors, [0,1,2], iris.target_names):
	    plt.scatter(x[y==i, 0], x[y==i, 1], color=color, label=target)

	plt.show()
	    
	#Making predictions - Clean this up!
	predictions = clf.predict(x)
	print(np.unique(predictions, return_counts=True)) #50 classified as 0, 53 classified as 1, 47 classified as 2
	# [0, 1, 2] equivalent to ['setosa' 'versicolor' 'virginica']

elif args.rbf:
	#Radial basis function (SBF) or Gaussian kernel 
	clf = svm.SVC(kernel='rbf', C=1, gamma='auto').fit(x,y) #Higher gammas can result in overfitting

	#Min and max for the first feature 
	x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1

	#Min and max for the second feature 
	y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

	#step size in the mesh 
	h = (x_max / x_min)/100

	#Make predictions for each of the points 
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	z = z.reshape(xx.shape)

	#Plotting using contourf function
	plt.contourf(xx, yy, z, cmap=plt.cm.Accent, alpha=0.5)

	#Plot training points 
	colors = ['red', 'green', 'blue']
	for color, i, target in zip(colors, [0,1,2], iris.target_names):
	    plt.scatter(x[y==i, 0], x[y==i, 1], color=color, label=target)
	plt.show()

	#Making predictions 
	predictions = clf.predict(x)
	print(np.unique(predictions, return_counts=True))
elif args.poly:
	#Polynomial kernel
	clf = svm.SVC(kernel='poly', degree=4, C=1, gamma='auto').fit(x,y)

	#Min and max for the first feature 
	x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1

	#Min and max for the second feature 
	y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

	#step size in the mesh 
	h = (x_max / x_min)/100

	#Make predictions for each of the points 
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	z = z.reshape(xx.shape)

	#Plotting using contourf function
	plt.contourf(xx, yy, z, cmap=plt.cm.Accent, alpha=0.5)

	#Plot training points 
	colors = ['red', 'green', 'blue']
	for color, i, target in zip(colors, [0,1,2], iris.target_names):
	    plt.scatter(x[y==i, 0], x[y==i, 1], color=color, label=target)
	plt.show()
else:
	print('Kernel not specified!')

