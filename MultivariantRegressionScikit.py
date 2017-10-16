# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 17:25:01 2017

@author: Chinmay Rawool
"""

import pandas as pd

df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')

df.head()

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].as_matrix())

print (X)

est = sm.OLS(y, X).fit()

est.summary()


y.groupby(df.Make).mean()
y.groupby(df.Make).median()