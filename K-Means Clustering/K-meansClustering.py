# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#library import
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



#dataset import
dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:,3:5].values


#K-Means model setup & WCSS 

from sklearn.cluster import KMeans
"""
wcss_list = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss_list.append(kmeans.inertia_)
print(kmeans.inertia_)


plt.plot(range(1,11),wcss_list)
plt.xlabel("Cluster Number")
plt.ylabel("WCSS Values")
plt.title("Elbow")
plt.show() #elbow graph shows us which n_cluster number for suitable for our data
"""



#K-means model setup with optimum n_cluster number
kmeans = KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(X)


#cluster graph

plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c="blue",label="Class 1")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c="green",label="Class 2")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c="red",label="Class 3")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c="purple",label="Class 4")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c="yellow",label="Class 5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="black",label="Centroids")
plt.legend()
plt.title("Mall Customer Segmentation")
plt.xlabel("Salary")
plt.ylabel("Score")
plt.show()


print(kmeans.cluster_centers_)
