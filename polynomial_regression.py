#!/usr/bin/env python

#Fitting data to a polynomial regression model in 3D space - Supervised Learning

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse as ap

parser = ap.ArgumentParser()
parser = ap.ArgumentParser(prog = 'Polynomial Regression Example', description='Explore training and plotting a polynomial model')
parser.add_argument("-d", "--degree", help="degree of polynomial", type=int, default=2, required=False)
args = parser.parse_args()

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
degree=args.degree
polynomial_features= PolynomialFeatures(degree=degree)
x_poly = polynomial_features.fit_transform(x)

feature_list = polynomial_features.get_feature_names(['x', 'y'])
#print(feature_list) #Somehow use feature_list to calculate z for nth degree polynomial instead of hardcode

#apply linear regression 
model = LinearRegression()
model.fit(x_poly, y)

#Calculating z from x and y of model - ideally, the feature list could be used to build functions nth degree polynomials
if degree==2:
	z = lambda x,y: (model.intercept_ + (model.coef_[1] * x) + (model.coef_[2] * y) + (model.coef_[3] * x**2) + (model.coef_[4] * x*y) + (model.coef_[5] * y**2))
elif degree==3:
	z = lambda x,y: (model.intercept_ + (model.coef_[1] * x) + (model.coef_[2] * y) + (model.coef_[3] * x**2) + (model.coef_[4] * x*y) + (model.coef_[5] * y**2) + (model.coef_[6] * x**3) + (model.coef_[7] * (x**2) * y) + (model.coef_[8] * (y**2) * x)  + (model.coef_[9] * y**3))

#Plotting plane
ax.plot_surface(x_surf, y_surf, z(x_surf, y_surf), rstride=1, cstride=1, color='None', alpha=0.4)
plt.show()