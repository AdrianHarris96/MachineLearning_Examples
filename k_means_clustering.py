#!/usr/bin/env python

#K means clustering - unsupervised learning - grouping with unlabeled data

#Kmeans example - Clothing Sizes
import numpy as np
import pandas as pd

df = pd.read_csv('/Users/adrianharris/Downloads/BMX_G.csv') #This can be provided as an input

#Checking for missing values 
df.isnull().sum()

#Two columns to pull based on input 

#Removing nulls from the two relevant columns 
df.dropna(subset=['bmxleg', 'bmxwaist'], inplace=True)

df.isnull().sum() #Checking those columns to ensure all is good

import matplotlib.pyplot as plt
plt.scatter(df['bmxleg'], df['bmxwaist'], c='r', s=2)
plt.xlabel("leg length (cm)")
plt.ylabel("waist circum. (cm)")
plt.show()

#Clustering with k_means
from sklearn.cluster import KMeans

k=2 #This can be provided as input
feat_matrix = np.array(list(zip(df['bmxleg'], df['bmxwaist'])))

kmeans = KMeans(n_clusters=k)
kmeans = kmeans.fit(feat_matrix)
labels = kmeans.predict(feat_matrix)
centroids = kmeans.cluster_centers_

#Map colors 
c = ['b', 'r', 'y', 'g', 'c', 'm']
colors = [c[i] for i in labels]

plt.scatter(df['bmxleg'], df['bmxwaist'], c=colors, s=2)
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



