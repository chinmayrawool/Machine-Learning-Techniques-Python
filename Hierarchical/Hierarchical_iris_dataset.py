# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:30:26 2017

@author: Chinmay Rawool
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import random

from sklearn.cluster import k_means
from sklearn.cluster import KMeans
from sklearn import datasets
from numpy import linalg as LA

def d_PCA(x, corr_logic):
    columnMean = x.mean(axis=0)
    columnMeanAll = np.tile(columnMean, reps=(x.shape[0], 1))
    xMeanCentered = x - columnMeanAll

    # use Non Standard data or standardized mean_centered data
    if not corr_logic:
        dataForPca = x
    else:
        dataForPca = xMeanCentered

    # get covariance matrix of the data
    covarianceMatrix = np.cov(dataForPca, rowvar=False)

    # eigendecomposition of the covariance matrix
    eigenValues, eigenVectors = LA.eig(covarianceMatrix)
    II = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[II]
    eigenVectors = eigenVectors[:, II]

    # get scores
    pcaScores = np.matmul(dataForPca, eigenVectors)

    # collect PCA results
    pcaResults = {'data': x,
                  'covarianceMatrix':covarianceMatrix,
                   'mean_centered_data': xMeanCentered,
                   'PC_variance': eigenValues,
                   'loadings': eigenVectors,
                   'scores': pcaScores}

    return pcaResults


def Hierarchical(Q1, Q2, S3, Q5):
    z = [] 
    Q2 = np.array(Q2)
    #Q5 = np.array(Q5)
    for i in range(len(Q2)):
        z.append(np.sqrt(((Q2[:,0] - Q2[i,0])**2) + ((Q2[:,1] - Q2[i,1])**2)))
    
    
    z1 = np.stack(z)
    min1 = min(z1[np.nonzero(z1)])
    
    minidx = np.where(z1 == min1)
    minidx2 = np.stack(minidx)
    minidx3 = minidx2[:,0]
    
    
    
    S1 = [Q1[minidx3[0]], Q1[minidx3[1]]]
    S2 = np.stack(S1)
    centx = np.mean(S2[:,0])
    centy = np.mean(S2[:,1])
    A1 = [centx, centy]
    S3.append([S1])
    #print S2
    
    #Q2[minidx3[0]] = A1
    
    #Q2 = np.delete(Q2, minidx3[1], axis=0)
    
    Q2 = np.delete(Q2, [minidx3[0], minidx3[1]], axis=0)
    #Q2 = np.delete(Q2, (minidx3[1] - 1), axis=0)
    
    Q2 = list(np.vstack([Q2, A1]))
    Q5 = list(np.vstack([Q5, S2]))
    #print Q5
    
    Q1.append(Q2)
    return Q1, Q2, Q5, min1

def main():
    #data initialization
#    np.random.seed(5)
#
#    iris = datasets.load_iris()
#    X = iris.data
#    feature_names = iris.feature_names
#    y = iris.target
#    target_names = iris.target_names
#
#    X1 = X[:,0:3]
    Y = pd.read_csv('D:\Coding\Python\ML in class\quiz3-chinmayrawool\SCLC_study_output_filtered_2.csv', sep=',',header=0);
    Y
    #D:\Coding\Python\ML in class\quiz3-chinmayrawool\SCLC_study_output_filtered.csv

    #Input data
    data = np.array(Y)
    new_data = np.array(data[:,1:],dtype=float)
    new_data

    #implement PCA
    dataForAnalysis = new_data #np.column_stack((x1, x2))

    #without Standard give False
    useCorr = False
    myPCAResults = d_PCA(x=dataForAnalysis, corr_logic=useCorr)
    X1 = new_data

    #Maintain Copy of X1
    Q1 = []
    Q2 = []
    Q5 = []
    for i in range(len(X1)):
        Q1.append(X1[i])
        Q2.append(X1[i])
        Q5.append(X1[i])
    
    threshold = 1.27279
    min2 = []
    min2.append(0)
    index = 0
    S3 = []
    count=1
    while True:
        Q1, Q2, Q5, min1 = Hierarchical(Q1, Q2, S3, Q5)
        if(count==1):
            first_pair = 
        min2.append(min1)
        if len(Q2) == 1:
            break
    Q1.append(Q2)
    while True:    
        if min2[index] < threshold:
            index = index + 1
                #print index
        else:
            nofclusters = len(Q1[(index + len(X1))])
            print nofclusters
            break
    dist = []    
    for i in range(len(X1)):
        dist.append(np.sqrt((X1[i,0]**2) + (X1[i,1]**2)))
    
    fig = plt.figure()
    plot_1 = fig.add_subplot(1,1,1)
    plot_1.scatter(dist, min2)
    fig.show()


if __name__ == '__main__':
    main()
