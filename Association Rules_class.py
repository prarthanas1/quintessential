# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 22:59:23 2018

@author: Prarthana Saikia
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

dd = pd.read_excel(r"C:\Users\Prarthana Saikia\Desktop\Praxis\Machine Learning\Store data.xlsx")

dd['cnt']=1

groups=dd.groupby(['CASH_MEMO','PRODUCT'])['cnt'].sum()
groups_unstack=dd.groupby(['CASH_MEMO','PRODUCT'])['cnt'].sum().unstack().fillna(0)  #by default, level of unstack is -1 


def recode(x):
    if x<=0:
        return 0
    if x>0:
        return 1
basket_bin = groups_unstack.applymap(recode)
basket_rules = apriori(basket_bin,min_support=0.01, use_colnames=True)
b=association_rules(basket_rules, metric='lift', min_threshold=2).sort_values(by='lift', ascending=False)

b["antecedant_len"] = b["antecedants"].apply(lambda x: len(x))

#rules[ (rules['antecedant_len'] >= 2) &
       #(rules['confidence'] > 0.75) &
       #(rules['lift'] > 1.2) ]