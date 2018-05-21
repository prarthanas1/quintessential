# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:21:42 2018

@author: Prarthana Saikia
"""

import numpy as np
import pandas as pd

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus 


#reading the file

social_data=pd.read_csv(r"C:\Users\Prarthana Saikia\Desktop\Praxis\Machine Learning\Machine Learning A-Z Template Folder\decision Tree\Decision_Tree_Classification\Social_Network_Ads.csv")

dummy=pd.get_dummies(social_data['Gender'])
social_data=social_data.drop('Gender',axis=1)
social_data=pd.concat([social_data,dummy],axis=1)


cols= social_data.columns.tolist()
columnsTitles = ['User ID', 'Age', 'EstimatedSalary','Female','Male','Purchased']
social_data=social_data.reindex(columns=columnsTitles)


X=social_data.iloc[:,1:5].values
y=social_data.iloc[:,5].values
#%%



#Data Preprocessing
from sklearn.cross_validation import train_test_split
#X and y are arrays
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  #already we have fitted the values in the data

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
model_tree=DecisionTreeClassifier(criterion='entropy',max_depth=3)
model_tree.fit(X_train,y_train)
print(model_tree)


dot_data=StringIO()

export_graphviz(model_tree,out_file=dot_data,filled=True,rounded=True,special_characters=True,proportion=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

#%%

# Predicting the Test set results
y_pred = model_tree.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,cohen_kappa_score, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cks=cohen_kappa_score(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)

#%%
# making the ROC curve
pred_prob =model_tree.predict_proba(X=X_test)   #find predicted probabilietes of input variables of test set
threshold_vals = np.linspace(0.0,1.0,num=10)
from sklearn.metrics import roc_curve
tpr = []
fpr = []
for th in threshold_vals:
       y_cap = np.where(pred_prob[:,0]<=th, 1,0)  
       cm = confusion_matrix(y_test,y_cap)  
       #print(cm)
       tpr.append(cm[1,1]/sum(cm[1,:]))
       fpr.append(cm[0,1]/sum(cm[0,:]))


       
import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.show()
tpr1, fpr1, th = roc_curve(y_test, pred_prob[:,0])
plt.plot(fpr1, tpr1)
