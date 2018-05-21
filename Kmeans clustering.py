# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 01:01:42 2018

@author: Prarthana Saikia
"""

import pandas as pd
from scipy import stats
from scipy.stats import zscore
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


dd=pd.read_table(r"C:\Users\Prarthana Saikia\Desktop\Praxis\Machine Learning\ML class\auto.txt",sep='\s+',header=None,na_values="?")
y=dd.head()
dx=pd.DataFrame(dd.values,columns=['mpg','cyl','disp','hp','wt','accel','year','origin','name'])

a=dx.isnull().sum(axis=0)
print(a)

#six null values are there so we can drop the rows which contain them

cleaned=dx.dropna()


#cannot work on categorical data so drop the last column

dx_temp=cleaned.drop('name',axis=1)


#standarduzed the data
dx_temp_standardized=dx_temp.apply(zscore)

#choosing the optimal number of clusters
#inertia is the sum of squared error for each cluster. smaller the inertia, denser the cluster

SS = []
for k in range (1,10):
    km = KMeans(n_clusters=k,random_state=1)
    ss= km.fit(dx_temp).inertia_
    SS.append(ss)
    print( "k:",k, " cost:", ss)

plt.plot(range(1,10),SS)
plt.show()                      #k=3 can be taken

from sklearn.metrics import silhouette_score
km= KMeans(n_clusters=3,n_init=40)
fit = km.fit(dx_temp)
fit.cluster_centers_
silhouette_score(X=dx_temp,labels=fit.labels_)

from sklearn.metrics import silhouette_score
km= KMeans(n_clusters=4,n_init=40)
fit = km.fit(dx_temp)
fit.cluster_centers_
silhouette_score(X=dx_temp,labels=fit.labels_)


a=dx_temp.iloc[:,0:2].values
s=4
plt.scatter(a[fit.labels_==0,0], a[fit.labels_==0,1], c = 'blue',s=s)
plt.scatter(a[fit.labels_==1,0], a[fit.labels_==1,1], c = 'red',s=s)
plt.scatter(a[fit.labels_==2,0], a[fit.labels_==2,1], c = 'green',s=s)
plt.scatter(a[fit.labels_==3,0], a[fit.labels_==3,1], c = 'cyan',s=s)
plt.show()
