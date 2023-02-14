#!/usr/bin/env python

#Fitting data to a linear regression model in both 2D and 3D space - Supervised Learning

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston() #Loading boston house-price data 

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
print(highest_corr_indices)

corr_values = df.corr().abs().nlargest(3, 'MEDV').values[:,13] #Extracting correlation values from the last column using the values method
corr_values = list(corr_values) 
print(corr_values)

x1 = df[highest_corr_indices[1]] #Recall, highest correlation will be between MEDV and itself
x2 = df[highest_corr_indices[2]]

'''
#Visualizing the correlations via matplotlib
plt.scatter(x1, df['MEDV'], marker='o')
plt.xlabel(highest_corr_indices[1])
plt.ylabel('MEDV')
plt.show()

plt.scatter(x2, df['MEDV'], marker='o')
plt.xlabel(highest_corr_indices[2])
plt.ylabel('MEDV')
plt.show()
'''
#Generating 3D plots 
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

quit()

#Training the model 
#Generate two dataframes 
x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT', 'RM'])
y = df['MEDV']

#Split dataset: 70% for training and 30% for testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)

#Start training 
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(x_train, y_train)

price_prediction = model.predict(x_test)
print(price_prediction)

#Assess the fit of the model 
print('R-squared: %.4f' % model.score(x_test, y_test))

#Scatterplot of actual vs. predicted price 
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, price_prediction)
print(mse)

plt.scatter(y_test, price_prediction)
plt.xlabel("actual prices")
plt.ylabel("predicted prices")
plt.title("predicted vs. actual")
plt.show()

#Extracting the model's intercept and coefficients
print(model.intercept_)
print(model.coef_)

#Prediction for an LSTAT of 30 and RM of 5
print(model.predict([[30,5]]))

