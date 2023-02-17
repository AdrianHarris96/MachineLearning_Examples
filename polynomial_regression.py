#!/usr/bin/env python

#Fitting data to a polynomial regression model in 3D space - Supervised Learning

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse as ap

#Add degree as an argument with a default=2
'''
parser = ap.ArgumentParser()
parser = ap.ArgumentParser(prog = 'Polynomial Regression Example', description='Explore training and plotting a polynomial model')
parser.add_argument("-y", "--plotxy", help="generate xy plot", action='store_true', required=False)
parser.add_argument("-z", "--plotxyz", help="generate xyz plot", action='store_true', required=False)
parser.add_argument("-t", "--training_plot", help="plot predicted vs. actual after training", action='store_true', required=False)
parser.add_argument("-p", "--plane", help="plot hyperplane", action='store_true', required=False)
args = parser.parse_args()
'''

#Remember: Setting the action to store_true allows the argument to be true in the code as long as it is invoked by the user

from sklearn.datasets import load_boston
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

dataset = load_boston() #Loading boston house-price data 

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['MEDV'] = dataset.target

#Generate dataframes
x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT', 'RM'])
y = df['MEDV']

fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x['LSTAT'], x['RM'], y, c='b')
ax.set_xlabel('LSTAT')
ax.set_ylabel('RM')
ax.set_zlabel('MEDV')

#Create meshgrid 
x_surf = np.arange(0, 40, 1)
y_surf = np.arange(0, 10, 1)

x_surf, y_surf = np.meshgrid(x_surf, y_surf)

#polynomial function
degree=2
polynomial_features= PolynomialFeatures(degree=degree)
x_poly = polynomial_features.fit_transform(x)
print(polynomial_features.get_feature_names(['x', 'y']))

quit()
#apply linear regression 
model = LinearRegression()
model.fit(x_poly, y)

#Calculating z from x and y of model
z = lambda x,y: (model.intercept_ + (model.coef_[1] * x) + (model.coef_[2] * y) + (model.coef_[3] * x**2) + (model.coef_[4] * x*y) + (model.coef_[5] * y**2))

#Plotting plane
ax.plot_surface(x_surf, y_surf, z(x_surf, y_surf), rstride=1, cstride=1, color='None', alpha=0.4)
plt.show()