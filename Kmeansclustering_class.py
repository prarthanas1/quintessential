# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 00:51:00 2018

@author: Prarthana Saikia
"""

import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

syn_data,cluster = make_blobs(n_samples = 1000, n_features = 2, centers = 4,random_state = 12345)

s=4
plt.scatter(syn_data[cluster==0,0],syn_data[cluster==0,1],c = 'blue',s=s)
plt.scatter(syn_data[cluster==1,0],syn_data[cluster==1,1],c = 'red',s=s)
plt.scatter(syn_data[cluster==2,0],syn_data[cluster==2,1],c = 'green',s=s)
plt.scatter(syn_data[cluster==3,0],syn_data[cluster==3,1],c = 'cyan',s=s)
plt.show()

km= KMeans(n_clusters=4,n_init=40)
fit = km.fit(syn_data)
fit.cluster_centers_

silhouette_score(X=syn_data,labels=fit.labels_)
s=4
plt.scatter(syn_data[fit.labels_==0,0], syn_data[fit.labels_==0,1], c = 'blue',s=s)
plt.scatter(syn_data[fit.labels_==1,0], syn_data[fit.labels_==1,1], c = 'red',s=s)
plt.scatter(syn_data[fit.labels_==2,0], syn_data[fit.labels_==2,1], c = 'green',s=s)
plt.scatter(syn_data[fit.labels_==3,0], syn_data[fit.labels_==3,1], c = 'cyan',s=s)
plt.show()

SS = []
for k in range (1,10):
    km = KMeans(n_clusters=k,n_init=100)
    ss= km.fit(syn_data).inertia_
    SS.append(ss)
    
plt.plot(range(1,10),SS)
plt.show()