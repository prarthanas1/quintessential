# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:01:40 2018

@author: Prarthana Saikia
"""

import numpy as np
import pandas as pd

dd=pd.read_table(r"C:\Users\Prarthana Saikia\Desktop\Praxis\Wholesale customers data.csv",sep=',',header='infer',na_values="?")
y=dd.head()

x=dd.drop(['Channel','Region'],axis=1)
import matplotlib.pyplot as plt
plt.scatter(x.iloc[:,0],x.iloc[:,1])

plt.title("title")
plt.xlabel("x-label")
plt.ylabel("y-label")
plt.show()

from scipy.cluster.hierarchy import linkage,dendrogram, cut_tree 
links=linkage(y=x,method='complete')

dendrogram(links)
plt.show()



c=cut_tree(links,height=50000)
print(c)
cluster_count=pd.DataFrame(c).iloc[:,0].value_counts()

# to see which point has gone to which cluster
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=8,affinity='euclidean',linkage='complete')
fit=hc.fit(x)   
clusterlabels=fit.labels_    #vector of clusters
print(clusterlabels.max())

c2=c.flatten

plt.scatter(x[c2==0,0],x[c2==0,1])
plt.scatter(x[c2==1,0],x[c2==1,1],c='red')
plt.show()
