# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:03:17 2017

@author: Chinmay Rawool
"""

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
from sklearn import linear_model
#from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets

diabetes = datasets.load_diabetes()
  
data_diabetes = np.array(diabetes.data[:,2])
target_diabetes = np.array(diabetes.target)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(data_diabetes, target_diabetes, color='red')
fig.show()

#randomize the data
data = np.column_stack((data_diabetes,target_diabetes))
random_array = np.copy(data)
np.random.shuffle(random_array)

#Select train and test data
X_train = random_array[:-20,0]
Y_train = random_array[:-20,1]

X_test = random_array[-20:,0]
Y_test = random_array[-20:,1]
    
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
#(422L,)
#(422L,)
#(20L,)
#(20L,)

# do linear regression using sklearn
lm_sklearn= linear_model.LinearRegression()
X_train = X_train.reshape((len(X_train), 1))
Y_train = Y_train.reshape((len(Y_train), 1))
X_test = X_test.reshape((len(X_test), 1))
Y_test = Y_test.reshape((len(Y_test), 1))
lm_sklearn.fit(X_train, Y_train)
y_hat_train = lm_sklearn.predict(X_train)
y_hat_test = lm_sklearn.predict(X_test)
    
#lm_sklearn_result = {}
#lm_sklearn_result['beta_0_hat'] = lm_sklearn.intercept_
#lm_sklearn_result['beta_1_hat'] = lm_sklearn.coef_
#lm_sklearn_result['R2'] = r2_score(train_y, y_hat)
#lm_sklearn_result['mean_squared_error'] = mean_squared_error(train_y, y_hat)
#lm_sklearn_result['y_hat'] = y_hat

#Plot the linear regression line with testing y vs testing x and predicted y vs testing x
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.scatter(X_train,Y_train,color='yellow',s=10)
ax.scatter(X_test, Y_test, color='red',s=50, label='testing_y vs testing_x') 
ax.scatter(X_test, y_hat_test, color='black',s=20, label='predicted_y vs testing_x')
ax.plot(X_train, y_hat_train, color='blue',linestyle='-', label='Linear regression line')
ax.legend()
fig.show()