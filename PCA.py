# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 20:16:20 2018

@author: Prarthana Saikia
"""

import pandas as pd
from sklearn.decomposition import PCA
from scipy.linalg import eig
import numpy as np

dd=pd.read_table(r"C:\Users\Prarthana Saikia\Desktop\Praxis\Machine Learning\ML class\auto.txt",sep='\s+',header=None,na_values="?")
y=dd.head()

a=dd.isnull().sum(axis=0)
print(a)

#remove the six values 

cleaned=dd.dropna()

#PCA

my_pca=PCA(n_components=2)
principalcomponents=my_pca.fit_transform(cleaned.iloc[:,:7])
principalDf = pd.DataFrame(data = principalcomponents
             , columns = ['principal component 1', 'principal component 2'])

#1. calculate covariance matrix

cov_mat= cleaned.iloc[:,:7].cov()
m=eig(cov_mat)

my_pca.explained_variance_ratio_

data = pd.concat([principalDf,cleaned.iloc[:,7:9]], axis = 1)


import matplotlib.pyplot as plt
pca = PCA().fit(cleaned.iloc[:,:7])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()