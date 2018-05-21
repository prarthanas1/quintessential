# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 00:33:43 2018

@author: Prarthana Saikia
"""

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus 

a=load_iris()

dd=pd.DataFrame(a['data'],columns=['sepal length','sepal width','petal length','petal width'])
df=pd.DataFrame(a['target'],columns=['Target Variable'])
work=pd.concat([dd,df],axis=1)

X=work.iloc[:,0:4].values
y=work.iloc[:,4].values

from sklearn.tree import DecisionTreeClassifier
model_tree=DecisionTreeClassifier(criterion='entropy',max_depth=3)
model_tree.fit(X,y)
print(model_tree)

dot_data=StringIO()

export_graphviz(model_tree,out_file=dot_data,filled=True,rounded=True,special_characters=True,proportion=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())