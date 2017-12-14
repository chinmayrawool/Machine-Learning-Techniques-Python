# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:32:14 2017

@author: Chinmay Rawool
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
#from sklearn.decomposition import PCA




def do_PCA(x, corr_logic):
    columnMean = x.mean(axis=0)
    columnMeanAll = np.tile(columnMean, reps=(x.shape[0], 1))
    xMeanCentered = x - columnMeanAll

    # use mean_centered data or standardized mean_centered data
    if not corr_logic:
        dataForPca = x
    else:
        dataForPca = xMeanCentered

    # Covariance Matrix of dataForPca
    coMatrix = np.cov(dataForPca, rowvar=False)

    # Calculate the Eigen values and Eigen vectors
    eigVal, eigVec = LA.eig(coMatrix)
    sorted_value = eigVal.argsort()[::-1]
    eigVal = eigVal[sorted_value]
    eigVec = eigVec[:, sorted_value]

    # Calculate the Pca scores
    scores = np.matmul(dataForPca, eigVec)

    # collect all the results
    results = {'data': x, 'mean_centered_data': xMeanCentered, 'PC_variance': eigVal, 'loadings': eigVec, 'scores': scores}

    return results

def linear_reg(x,y):
    n = len(x)

    x_bar = np.mean(x)
    y_bar = np.mean(y)

    S_yx = np.sum((y - y_bar) * (x - x_bar))
    S_xx = np.sum((x - x_bar)**2)

    # calculate beta_0 and beta_1
    beta_1_hat = S_yx / S_xx 
    beta_0_hat = y_bar - beta_1_hat * x_bar

    # calculate y_hat 
    y_hat = beta_0_hat + beta_1_hat * x
    #r = y - y_hat
    #sigma_hat = np.sqrt(sum(r**2) / (n-2))
    
    linreg_results = {'beta_1_hat':beta_1_hat, 'beta_0_hat':beta_0_hat, 'y_hat':y_hat}
    
    return linreg_results


def main():

    # ----------------------------------------------------------------
    # Q.1   Part 1
    # ----------------------------------------------------------------

    # 1. Input data
    filePath = 'D:\Coding\Python\ML in class\Homework 1\linear_regression_test_data.csv'
    input_data = pd.read_csv(filePath, sep=',',header=0);
    input_data

    x = np.array(input_data['x'])
    y = np.array(input_data['y'])
    y_theoretical = np.array(input_data['y_theoretical'])
    
    data = np.array(input_data)
    new_data = np.array(data[:,1:],dtype=float)
    new_data

    # 2. do PCA on x and y
    dataForPCA = np.column_stack((x, y)) #np.column_stack((x1, x2))

    useCorr = True
    myPCAResults = do_PCA(x=dataForPCA, corr_logic=useCorr)

    #PC1_data = myPCAResults['scores'][:,0]
    
    #-----------
    # PC1 axis is calculated from the Eigen Vectors of the covariance matrix
    #-----------
    
    # 3. visualize the raw data: y vs x, y-theoretical vs x, y vs PC1
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x,y, color='blue',label='y vs x')
    ax.scatter(x, y_theoretical, color='red',label='y theoretical vs x')
    k=3
    ax.plot([0,(-1)*k*myPCAResults['loadings'][0,0]], [0,(-1)*k*myPCAResults['loadings'][1,0]], color='black',linewidth=3, label='PC1 axis')
    ax.legend()
    fig.show()
    
    # -------------------------------------------------------------------------------
    # Q.1   Part 2    Linear Regression x is independent and y is dependent variable
    # -------------------------------------------------------------------------------
   
    n = len(x)
    
    linearRegResults = linear_reg(x,y)
    print("Beta 0 value:",linearRegResults['beta_0_hat'])
    print("Beta 1 value:",linearRegResults['beta_1_hat'])
    
#    ('Beta 0 value:', 0.024525025871305339)
#    ('Beta 1 value:', 1.9340058850010582)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, color='blue', label='y vs x')
    ax.scatter(x, y_theoretical, color='red',label='y theoretical vs x')
    ax.plot(x, linearRegResults['y_hat'], color='green', label='Linear regression line')
    k=3
    ax.plot([0,(-1)*k*myPCAResults['loadings'][0,0]], [0,(-1)*k*myPCAResults['loadings'][1,0]], color='black',linewidth=3, label='PC1 axis')
    ax.legend()
    fig.show()

if __name__ == '__main__':
    main()
