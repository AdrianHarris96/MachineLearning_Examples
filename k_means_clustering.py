#!/usr/bin/env python

#K means clustering - unsupervised learning - grouping with unlabeled data

#Example input: ./k_means_clustering.py -i /Users/adrianharris/Downloads/BMX_G.csv -c1 bmxleg -c2 bmxwaist

#Kmeans example - Clothing Sizes
import numpy as np
import pandas as pd
import argparse as ap

parser = ap.ArgumentParser()
parser = ap.ArgumentParser(prog = 'K-means Clustering', description='K-means clustering for clothing sizes')
parser.add_argument("-i", "--input", help="path to input BMX_G CSV", required=True)
parser.add_argument("-c1", "--col1", help="1st column to extract from dataframe", required=True)
parser.add_argument("-c2", "--col2", help="2nd column to extract from dataframe", required=True)
args = parser.parse_args()

input_file = args.input
df = pd.read_csv(input_file)

#Checking for missing values 
df.isnull().sum()

#Two columns to pull based on input 
x = args.col1
y = args.col2

#Removing nulls from the two relevant columns 
df.dropna(subset=[x, y], inplace=True)

df.isnull().sum() #Checking those columns to ensure all is good

import matplotlib.pyplot as plt
plt.scatter(df[x], df[y], c='r', s=2)
plt.xlabel(x)
plt.ylabel(y)
plt.show()

quit()

#Clustering with k_means
from sklearn.cluster import KMeans

k=2 #This can be provided as input
feat_matrix = np.array(list(zip(df[x], df[y])))

kmeans = KMeans(n_clusters=k)
kmeans = kmeans.fit(feat_matrix)
labels = kmeans.predict(feat_matrix)
centroids = kmeans.cluster_centers_

#Map colors 
c = ['b', 'r', 'y', 'g', 'c', 'm']
colors = [c[i] for i in labels]

plt.scatter(df[x], df[y], c=colors, s=2)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=100, c='black')
plt.show()

#Optimal K
from sklearn import metrics

#Finding the optimal K using the silhoutte mean 
silhoutte_averages = []
min_k = 2

#Loop from minimum k to maximum k (num of rows)
for k in range(min_k, 10): #max_k can be taken as input
    kmean = KMeans(n_clusters=k).fit(feat_matrix)
    score = metrics.silhouette_score(feat_matrix, kmean.labels_)
    print('Silhoutte coef. for k = {0} is {1}'.format(k, score))
    silhoutte_averages.append(score) #append to list to later plot



