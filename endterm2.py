# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:47:07 2018

@author: Prarthana Saikia
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import zscore
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


bank_data1=pd.read_csv(r"C:\Users\Prarthana Saikia\Desktop\Praxis\Machine Learning\ML class\codes for exam\bank_data.csv")
bank_data1.head()



dx_temp_standardized=bank_data1.apply(zscore)


SS = []
for k in range (1,10):
    km = KMeans(n_clusters=k,random_state=1)
    ss= km.fit(dx_temp_standardized).inertia_
    SS.append(ss)
    print( "k:",k, " cost:", ss)

plt.plot(range(1,10),SS)
plt.show()                      #k=3 can be taken

from sklearn.metrics import silhouette_score
km= KMeans(n_clusters=6,n_init=40)
fit = km.fit(dx_temp_standardized)
fit.cluster_centers_
silhouette_score(X=dx_temp_standardized,labels=fit.labels_)

from sklearn.metrics import silhouette_score
km= KMeans(n_clusters=6,n_init=40)
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