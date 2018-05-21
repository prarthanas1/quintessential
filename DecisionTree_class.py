# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 12:36:36 2018

@author: Prarthana Saikia
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus   #convert model output into graph format

bankloan=pd.read_csv(r"C:\Users\Prarthana Saikia\Desktop\Praxis\Machine Learning\ML class\bankloan.csv",na_values='#NULL!')
bankloan.head()

#predict the cases for 150 remaining new customers

bankloan_known=bankloan[~bankloan.default.isnull()] #bankloan default that dont contain null

#%%
bankloan_known['default']=bankloan_known['default'].astype('int')
bankloan_unknown=bankloan[bankloan.default.isnull()]

model_tree=DecisionTreeClassifier(criterion='entropy',max_depth=3)
model_tree.fit(X=bankloan_known.drop('default',axis=1),y=bankloan_known['default'])
print(model_tree)
#%%
dot_data=StringIO()

export_graphviz(model_tree,out_file=dot_data,filled=True,rounded=True,special_characters=True,proportion=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


