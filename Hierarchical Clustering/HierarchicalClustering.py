# -*- coding: utf-8 -*-
"""
Created on Fri May 16 16:38:22 2025

@author: omer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")

X = dataset.iloc[:,3:5].values
"""
#Optimum Cluster Sayısının Bulunabilmesi İçin Dendrogram Kullanılması

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X,method="ward"))
plt.title("Dendrogram Ward")
plt.xlabel("Customers")
plt.ylabel("Distances")

"""

#Hierarchical Modelinin Kurulması
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=3, affinity="euclidean",linkage="ward")


y_hc = hc.fit_predict(X)

#Clusterların Plot Olarak Çizilmesi
plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], s=100, c="blue", label="Class 1")
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], s=100, c="green", label="Class 2")
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], s=100, c="red", label="Class 3")
"""
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], s=100, c="purple", label="Class 4")
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], s=100, c="yellow", label="Class 5")
"""
plt.legend()
plt.title("Müşteri Segmentasyonu - HC - 3 Cluster")
plt.xlabel("Yıllık Gelir")
plt.ylabel("Harcama Skoru")
plt.show()
