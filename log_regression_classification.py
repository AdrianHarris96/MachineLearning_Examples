#!/usr/bin/env python

#Classification with logistic regression

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import argparse as ap

parser = ap.ArgumentParser()
parser = ap.ArgumentParser(prog = 'Logistic Regression for Classification Example', description='Use log regression to classify tumor as benign or malignant based on feature(s)')
parser.add_argument("-s", "--single_feat", help="single feature for classification", action="store_true", required=False)
parser.add_argument("-a", "--all_feat", help="all features for classification", action="store_true", required=False)
args = parser.parse_args()

from sklearn.datasets import load_breast_cancer 
cancer = load_breast_cancer() #Breats cancer diagnostic dataset 
#print(cancer.DESCR)

if args.single_feat:
	col = input('Enter the desired col (0-index): ')
	col = int(col)
	#Training using on feature - mean radius
	x = cancer.data[:, col] #all rows, single col
	y = cancer.target #status - malignant or benign 

	'''
	print(x)
	print(y)
	'''

	feat_name = cancer.feature_names[col] #name of col to be used to label the axis later


	colors = {0: 'red', 1: 'blue'} #dictionary: red is malignant and blue is benign

	'''
	#Simply plot points
	plt.scatter(x, y, facecolors='none', edgecolors=pd.DataFrame(cancer.target)[0].apply(lambda x: colors[x]), cmap=colors) #Applying the first classification status in that dataframe col to corresponding color in the dictionary (hence, the 0 and 1)

	plt.xlabel(feat_name)
	plt.ylabel("result")

	red = mpatches.Patch(color = 'red', label = 'malignant')
	blue = mpatches.Patch(color = 'blue', label = 'benign')

	plt.legend(handles=[red, blue], loc=1)
	plt.show()

	'''

	#Training with single feature
	from sklearn import linear_model

	log_regress = linear_model.LogisticRegression()

	#Train model from mean radius data 
	log_regress.fit(X = np.array(x).reshape(len(x), 1), y = y)

	#Extracting intercept and beta coefficent 
	inter = log_regress.intercept_[0]
	#print(log_regress.intercept_)

	c = log_regress.coef_[0][0]
	#print(log_regress.coef_)

	#Plotting sigmoid curve 
	def sigmoid(x):
	    return (1 / (1 + np.exp(-(inter + (c * x)))))

	x1 = np.arange(0, 30, 0.01)
	y1 = [sigmoid(n) for n in x1]

	#Plot scatter 
	plt.scatter(x, y, facecolors='none', edgecolors=pd.DataFrame(cancer.target)[0].apply(lambda x: colors[x]), cmap=colors)

	plt.plot(x1, y1)

	plt.xlabel(feat_name)
	plt.ylabel("prob.")

	plt.show()

	predict_val = input('Input value to predict: ')
	if predict_val == '':
		print('No input')
	else:
		predict_val = int(predict_val)
		#Making predictions 
		test = [predict_val] #Change this number to toy around with the predictions 
		test = np.array(test)
		test = test.reshape(-1, 1) #input needs to be a rank 2 np array, hence the code above to do the conversion
		print('[Prob. of 0, Prob. of 1]:', log_regress.predict_proba(test)) #output is [probability of 0 "malignant" probability of 1 "benign"]
		print('Class:', log_regress.predict(test)) #Simply prints the result, in this case, malignant
elif args.all_feat:
	#Training model using all features 
	from sklearn.model_selection import train_test_split 
	train_set, test_set, train_labels, test_labels = train_test_split(cancer.data, cancer.target, test_size = 0.25, random_state = 1, stratify = cancer.target)

	#test_size indicates the percentage allocated to test - 25% to test, 75% to train
	#random_state is simply the seed
	#stratify indicates what to randomize on 

	#Training the model 
	from sklearn import linear_model 
	x = train_set[:, 0:30]
	y = train_labels

	log_regress = linear_model.LogisticRegression(max_iter=5000) #change the max iterations 
	log_regress.fit(X = x, y = y)

	#Extract intercept and 30 coefficients
	inter = log_regress.intercept_
	c = log_regress.coef_

	#Testing the model
	prediction_probs = pd.DataFrame(log_regress.predict_proba(X=test_set)) #predictions made via test_set

	prediction_probs.columns = ["Malignant", "Benign"]

	#Put the predictions themselves in a dataframe
	predictions = log_regress.predict(X=test_set)
	pred_df = pd.DataFrame(predictions)
	pred_df.columns = ["Predictions"]

	#Concatenate 
	result_df = pd.concat([prediction_probs, pred_df], axis = 1) #Stitch dataframes to together (glue columns in this case using axis=1)
	result_df['Original Result'] = test_labels #Add the original classifications 
	print(result_df)
else:
	print('Run log_regression_classification.py with flag')


	

