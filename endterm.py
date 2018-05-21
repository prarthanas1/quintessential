# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:23:27 2018

@author: Prarthana Saikia
"""

import numpy as np
import pandas as pd


from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

bank_data1=pd.read_csv(r"C:\Users\Prarthana Saikia\Desktop\Praxis\Machine Learning\ML class\codes for exam\bank_data.csv")
bank_data1.head()

X= bank_data1.iloc[:,0:9].values.astype('int')
y=bank_data1.iloc[:,9].values.astype('int')

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from sklearn.tree import DecisionTreeClassifier
model_tree=DecisionTreeClassifier(criterion='entropy',max_depth=3)
model_tree.fit(X_train,y_train)

y_pred = model_tree.predict(X_test)
from sklearn.metrics import confusion_matrix,cohen_kappa_score, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(cohen_kappa_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))