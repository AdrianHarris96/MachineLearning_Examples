#!/usr/bin/env python

#Classification with logistic regression

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 

from sklearn.datasets import load_breast_cancer 
cancer = load_breast_cancer() #Breats cancer diagnostic dataset 
#print(cancer.DESCR)

#Training using on feature - mean radius
x = cancer.data[:, 0] #all rows, column 0 - mean radius
y = cancer.target #status - malignant or benign 

'''
print(x)
print(y)
'''

colors = {0: 'red', 1: 'blue'} #dictionary: red is malignant and blue is benign

plt.scatter(x, y, facecolors='none', edgecolors=pd.DataFrame(cancer.target)[0].apply(lambda x: colors[x]), cmap=colors) #Applying the first classification status in that dataframe col to corresponding color in the dictionary (hence, the 0 and 1)

plt.xlabel("mean radius")
plt.ylabel("result")

red = mpatches.Patch(color = 'red', label = 'malignant')
blue = mpatches.Patch(color = 'blue', label = 'benign')

plt.legend(handles=[red, blue], loc=1)
plt.show()
