# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:11:21 2018

@author: Prarthana Saikia
"""

from sklearn.datasets import make_blobs,make_circles
from scipy.cluster.hierarchy import linkage,dendrogram, cut_tree  #cut dendrogram at certain height
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

some_data,cluster=make_blobs(n_samples=20,n_features=2,random_state=12345,centers=2) #linear separablity
#some_data1,cluster1=make_circles(n_samples=100,random_state=12345)   #non linear separability

plt.scatter(some_data[:,0],some_data[:,1])



#plt.scatter(some_data1[:,0],some_data1[:,1])
#plt.show()

#for hierarchical clustering need linkage function

links=linkage(y=some_data,method='complete')

dendrogram(links)
plt.show()

c=cut_tree(links,height=10)
print(c)


# to see which point has gone to which cluster
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='complete')
fit=hc.fit(some_data)   
clusterlabels=fit.labels_    #vector of clusters
print(clusterlabels.max())



#inject outlier

some_data_outlier=np.vstack([some_data, np.array([30.098,40.234])]) #add extra row vertically

links_outlier=linkage(some_data_outlier,method='complete')

dendrogram(links_outlier)
plt.show()   #3rd cluster is outlier


#using some_data1 dataset

#links_circles=linkage(y=some_data1,method='complete')
#links_circles=linkage(y=some_data1,method='single')

#dendrogram(links_circles)
#plt.show()

#c1=cut_tree(links_circles,height=0.175)
#print(c1)

#trying to plot the circles
#c2=c.flatten()
c2=clusterlabels

plt.scatter(some_data[c2==0,0],some_data[c2==0,1])
plt.scatter(some_data[c2==1,0],some_data[c2==1,1],c='red')
plt.show()

#plt.scatter(some_data1[c1[:,0]==0,0],some_data1[c1[:,0]==0,1])
#plt.scatter(some_data1[c1[:,0]==1,0],some_data1[c1[:,0]==1,1],c='red')
#plt.show()
