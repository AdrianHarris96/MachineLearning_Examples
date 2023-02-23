#!/usr/bin/env python

#Manual implementation of KMeans Clustering 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

array = np.array([[1,1],
                 [2,2],
                 [2,3],
                 [1,4],
                 [3,3], 
                 [6,7],
                 [7,8],
                 [6,8],
                 [7,6],
                 [6,9],
                 [2,5],
                 [7,8],
                 [3,1],
                 [8,4],
                 [8,6],
                 [8,9]])

df = pd.DataFrame(array, columns = ['x', 'y'])
df

plt.scatter(df['x'], df['y'], c='r', s=18)
plt.show()

#Generating random centroids 
k = 3

pt_matrix = np.array(list(zip(df['x'], df['y']))) #matrix containing the points

#Generate k random centroids as a matrix 
cent_x = np.random.randint(np.min(pt_matrix[:,0]), np.max(pt_matrix[:,0]), size=k) #Creating x coordinates using the point matrix mins and maxs for x
cent_y = np.random.randint(np.min(pt_matrix[:,1]), np.max(pt_matrix[:,1]), size=k) #Creating y coordinates using the point matrix mins and maxs for y

#Represent the centroids in a matrix 
c_matrix = np.array(list(zip(cent_x, cent_y)), dtype=np.float64)
print(c_matrix)

#Plot scatter plot with points and centroids
plt.scatter(df['x'], df['y'], c='r', s=18)
plt.scatter(cent_x, cent_y, marker='*', c='g', s=160)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Manual implementation of k-means clustering algorithm
from copy import deepcopy

#Calculate euclidean distance between two points
def euclidean_dist(pt1, pt2, ax=1):
    return np.linalg.norm(pt1 - pt2, axis=ax)

c_previous = np.zeros(c_matrix.shape) #matrix of zeros with same dimensions as centroid matrix

clusters = np.zeros(len(pt_matrix)) #The cluster each point belongs to 

#Measuring the distance between the current centroids and previous centroids
dist_diff = euclidean_dist(c_matrix, c_previous)

colors = ['b', 'r', 'y', 'g', 'c', 'm'] #Simply colors for the clusters

#Loop as long as there is a difference between previous and current centroids
while dist_diff.any() != 0: #if any in that array are not 0, continue
    for i in range(len(pt_matrix)): #iterating through the rows
        distances = euclidean_dist(pt_matrix[i], c_matrix)
        
        cluster = np.argmin(distances) #returns the indices of the min values along an axis
        clusters[i] = cluster
        
        c_previous = deepcopy(c_matrix) #store the previous centroids
        
        #Find the new centroids by taking the average value 
        for i in range(k): #k being the number of clusters
            points = [pt_matrix[j] for j in range(len(pt_matrix)) if clusters[j] == i]
            if len(points) != 0:
                c_matrix[i] = np.mean(points, axis=0)
                
        dist_diff = euclidean_dist(c_matrix, c_previous)
        
        #Display how the centroids change in this while loop
        '''
        plt.scatter(c_matrix[:,0], c_matrix[:,1], s=100, marker='*', c='black')
        plt.show()
        '''
        
        
for i in range(k):
    points = np.array([pt_matrix[j] for j in range(len(pt_matrix)) if clusters[j] == i])
    if len(points) > 0:
        plt.scatter(points[:,0], points[:,1], s=10, c=colors[i])
    else: #One of the clusters may have no points
        print("Please regenerate centroids") #Write the code in a way in which it will re-run on its own
        
plt.scatter(points[:,0], points[:,1], s=10, c=colors[i])
plt.scatter(c_matrix[:,0], c_matrix[:,1], s=100, marker='*', c='black')

#Print the cluster each point belongs to 
for i, cluster in enumerate(clusters):
    print("Point " + str(pt_matrix[i]), "Cluster "+ str(int(cluster)))