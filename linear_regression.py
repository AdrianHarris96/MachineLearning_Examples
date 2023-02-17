#!/usr/bin/env python

#Fitting data to a linear regression model in both 2D and 3D space - Supervised Learning

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse as ap

parser = ap.ArgumentParser()
parser = ap.ArgumentParser(prog = 'Linear Regression Example', description='Explore training and plotting a linear model')
parser.add_argument("-y", "--plotxy", help="generate xy plot", action='store_true', required=False)
parser.add_argument("-z", "--plotxyz", help="generate xyz plot", action='store_true', required=False)
parser.add_argument("-t", "--training_plot", help="plot predicted vs. actual after training", action='store_true', required=False)
parser.add_argument("-p", "--plane", help="plot hyperplane", action='store_true', required=False)
args = parser.parse_args()

#Remember: Setting the action to store_true allows the argument to be true in the code as long as it is invoked by the user

from sklearn.datasets import load_boston

dataset = load_boston() #Loading boston house-price data 
#print(dataset.DESCR) #Description of each feature 

#Load the dataset into pandas dataframe 
df = pd.DataFrame(dataset.data, columns = dataset.feature_names) #feature_names are synonymous with columns for dataframe

df['MEDV'] = dataset.target #This will serve as the y-coordinate (dependent variable)
#df.head()

#Pairwise correlation of columns
corr_matrix = df.corr() #Similar to the method in R
#print(corr_matrix)

#Identifying top 3 correlations
highest_corr_indices = df.corr().abs().nlargest(3, 'MEDV').index #Extracting the index of the features with the highest corrleations
highest_corr_indices = list(highest_corr_indices) #Conversion to list
print("List of Top Correlated Columns")
print(highest_corr_indices)

corr_values = df.corr().abs().nlargest(3, 'MEDV').values[:,13] #Extracting correlation values from the last column using the values method
corr_values = list(corr_values)
print("Top Correlation Values ")
print(corr_values)

x1 = df[highest_corr_indices[1]] #Recall, highest correlation will be between MEDV and itself
x2 = df[highest_corr_indices[2]]

#Visualizing the correlations (2D) via matplotlib
if args.plotxy:
	plt.scatter(x1, df['MEDV'], marker='o')
	plt.xlabel(highest_corr_indices[1])
	plt.ylabel('MEDV')
	plt.show()

	plt.scatter(x2, df['MEDV'], marker='o')
	plt.xlabel(highest_corr_indices[2])
	plt.ylabel('MEDV')
	plt.show()

#Generating 3D plots
if args.plotxyz: 
	from mpl_toolkits.mplot3d import Axes3D

	#Specifications for the plot
	fig = plt.figure(figsize=(18,15))
	ax = fig.add_subplot(111, projection='3d')

	#Generating scatter plot
	ax.scatter(x1, x2, df['MEDV'], c='b')
	ax.set_xlabel(highest_corr_indices[1])
	ax.set_ylabel(highest_corr_indices[2])
	ax.set_zlabel('MEDV')
	plt.show()

#Training the model 
#Generate two dataframes 
x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT', 'RM']) #Two independent features in one dataframe
y = df['MEDV'] #Dependent feature in another

#Split dataset: 70% for training and 30% for testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)

#Start training 
from sklearn.linear_model import LinearRegression

#Fit to training data
model=LinearRegression()
model.fit(x_train, y_train)

#Use the predict() method to generate predictions based on the x_test
price_prediction = model.predict(x_test)
#print(price_prediction) 

#Assess the fit of the model 
print('R-squared: %.4f' % model.score(x_test, y_test)) #4 sig. figs.

#Scatterplot of actual vs. predicted price 
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, price_prediction)
print('Mean Squared Error (MSE): %.4f' % mse)

#Simple scatter of actual vs. predicted price after training
if args.training_plot:
	plt.scatter(y_test, price_prediction)
	plt.xlabel("actual prices")
	plt.ylabel("predicted prices")
	plt.title("predicted vs. actual")
	plt.show()

#Extracting the model's intercept and coefficients
print('Beta0 (Intercept): %.4f' % model.intercept_)
print('Beta1: {0}\nBeta2: {1}'.format(model.coef_[0], model.coef_[1]))

#Prediction of MEDV for an LSTAT of 30 and RM of 5
#print(model.predict([[30,5]]))

#Generate hyperplane
#Note: Only the training data is used here
if args.plane:
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure(figsize=(18,15))
	ax = fig.add_subplot(111, projection='3d')

	#Simple scatter
	ax.scatter(x['LSTAT'], x['RM'], y, c='b')
	ax.set_xlabel('LSTAT')
	ax.set_ylabel('RM')
	ax.set_zlabel('MEDV')

	#Create meshgrid 
	x_surf = np.arange(0, 40, 1) #x-coordinates for surface/plane
	y_surf = np.arange(0, 10, 1) #y-coordinates for surface/plane

	x_surf, y_surf = np.meshgrid(x_surf, y_surf)

	from sklearn.linear_model import LinearRegression
	model = LinearRegression()
	model.fit(x_train, y_train)

	#Calculate z based on x and y in model 
	z = lambda x_train,y_train: (model.intercept_ + model.coef_[0] * x_train + model.coef_[1] * y_train) #Simply using the linear model formula (B0 + B1x + B2y) to generate the z
	ax.plot_surface(x_surf, y_surf, z(x_surf, y_surf), rstride=1, cstride=1, color='None', alpha=0.4)
	plt.show()

