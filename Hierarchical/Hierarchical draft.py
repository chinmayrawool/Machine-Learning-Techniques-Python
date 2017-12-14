# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:31:20 2017

@author: Chinmay Rawool
"""

#import sys
#import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import random

from sklearn.cluster import k_means
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd


#np.set_printoptions(threshold = 'nan')

#data = np.genfromtxt('test.csv', dtype=float, delimiter=',', names=True) 
#
#reader = csv.reader(open('test.csv', 'r'), delimiter = ',')
#x = list(reader)
#result = np.array(x).astype('string')
#first_row = result[0,:]
#first_row = first_row[1:]
#first_row2 = first_row.astype(np.float)
#first_column = result[:,0]
#result1 = np.delete(result, 0, 1)
#result2 = np.delete(result1, 0, 0)
#result3 = result2.astype(np.float)
#
#X = result3

#print X
#for i in range(len(X)):
#    for j in range(len(X[0])):
#        Q1.append([X[i,j],first_row2[j]])

#z = []
#for i in range(len(X)):
#	for j in range(len(X[0])):
#		z.append(np.sqrt((X - X[i,j])**2 + (first_row2 - first_row2[j])**2))
#
#z1 = np.stack(z)
#min1 = min(z1[np.nonzero(z1)])
#minidx = np.where(z1 == min1)
#minidx2 = np.stack(minidx)
#minidx3 = minidx2[:,0]
#
#X2 = np.stack(X)
#X3 = X.ravel()
#
#S1 = [X3[minidx3[0]], X[minidx3[1],minidx3[2]]]
#S2 = [first_row2[minidx3[0]],first_row2[minidx3[2]]]
#
#centx = np.mean(S1)
#centy = np.mean(S2)


#data initialization
np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
feature_names = iris.feature_names
y = iris.target
target_names = iris.target_names

X1 = X[:,0:2]

#Data = [[0.9,0.9],[0.3,0.7],[0.35,0.65],[0.55,0.1],[0.9,0.2],[0.2,0.2],[0.9,0.7]]
#Data = np.array(Data)
#X1 = Data

Q1 = []
Q2 = []
Q5 = []
for i in range(len(X1)):
    Q1.append(X1[i])
    Q2.append(X1[i])
    Q5.append(X1[i])

#print 'Q2',Q2
#print 'Q1',Q1
#SQ1 = np.stack(Q1)

#z = []    
#for i in range(len(X1)):
#    z.append(np.sqrt(((X1[:,0] - X1[i,0])**2) + ((X1[:,1] - X1[i,1])**2)))
#
#z1 = np.stack(z)
#min1 = min(z1[np.nonzero(z1)])
#minidx = np.where(z1 == min1)
#minidx2 = np.stack(minidx)
#minidx3 = minidx2[:,0]
#
#
#
#S1 = [X1[minidx3[0]], X1[minidx3[1]]]
#S2 = np.stack(S1)
#centx = np.mean(S2[:,0])
#centy = np.mean(S2[:,1])
#A1 = [centx, centy]
#
##Q2[minidx3[0]] = A1
#
##Q2 = np.delete(Q2, minidx3[1], axis=0)
#
#Q2 = np.delete(Q2, [minidx3[0], minidx3[1]], axis=0)
##Q2 = np.delete(Q2, (minidx3[1] - 1), axis=0)
#
#Q2 = list(np.vstack([Q2, A1]))
#Q5 = list(np.vstack([Q5, S2]))
#
#Q1.append(Q2)



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

threshold = 1.27279
min2 = []
min2.append(0)
index = 0
S3 = []
while True:
    Q1, Q2, Q5, min1 = Hierarchical(Q1, Q2, S3, Q5)
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