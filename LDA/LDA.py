# -*- coding: utf-8 -*-
"""
Created on Sat Dec 09 19:45:30 2017

@author: Chinmay Rawool
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def calculate_S_within(X1,X2,mu_1,mu_2):
    S1 = np.zeros((19,19),dtype='float')
    S2 = np.zeros((19,19),dtype='float')
    for i in range(20):
        temp = X1[i,:] - mu_1 
        print temp
        temp = np.array(temp)
        temp=temp.reshape((19,1))
        temp_T = temp.T
        S1 = S1 + temp.dot(temp_T)
    for i in range(20):
        temp = X2[i,:] - mu_2
        print temp
        temp = np.array(temp)
        temp=temp.reshape((19,1))
        temp_T = temp.T
        S2 = S2 + temp.dot(temp_T) 
    
    print "S1 - ",S1
    print "S2 - ",S2

    #Calculate S within
    S_within = S1 + S2
    return S1,S2,S_within
    
def calculate_S_between(mu_1,mu_2,inputMat_mean):
    
    #Create a matrix of mean vectors
    mean_vectors = np.array([mu_1,mu_2])
    print "Mean Vectors - ", mean_vectors.shape
    
    S_between = np.zeros((19,19))
    for i,mean_vec in enumerate(mean_vectors):
        mean_vec = mean_vec.reshape(19,1) 
        S_between = np.add(S_between,(20 * (mean_vec - inputMat_mean).dot((mean_vec - inputMat_mean).T)))
    print "S_between: ", S_between
    return S_between

#----------------------
# Initialize the input matrix
#---------------------
inputFile = pd.read_csv("D:\Coding\Python\ML in class\LDA\SCLC_study_output_filtered_2.csv",header=0)
dataF = pd.DataFrame(inputFile)
inputMat = dataF.as_matrix()
inputMat = inputMat[:,1:]
inputMat = inputMat.astype('float')
print "inputmat: ", inputMat

#Separate the input matrix into X1 and X2
X1 = inputMat[:20,:]
#X1 = np.array(X1,dtype='float')
print "X1: ", X1
X2 = inputMat[20:,:]
print "X2: ", X2

#Calculate mean of X1 and X2
mu_1 = X1.mean(axis=0)
mu_2 = X2.mean(axis=0)
print "Mean of X1 - ",mu_1
print "Mean of X2 - ",mu_2

#Calculate S1,S2 and S_within
S1,S2,S_within = calculate_S_within(X1,X2,mu_1,mu_2)

#Calculate S between and overall mean of inputMat
inputMat_mean = np.mean(inputMat, axis=0)
inputMat_mean = inputMat_mean.reshape(19,1)

S_between = calculate_S_between(mu_1,mu_2,inputMat_mean)

#Calculate inverse of S_within
S_within = np.array(S_within, dtype='float')
S_within_inv = np.linalg.inv(S_within)
S_within_inv

#Calculate S_total
S_total = np.dot(S_within_inv,S_between)

eigen_val, eigen_vec = np.linalg.eig(S_total)
eigen_val
eigen_vec

eig_pairs = [(np.abs(eigen_val[i]), eigen_vec[:,i]) for i in range(len(eigen_val))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)


#Create a matrix containing first 3 highest Eigen Vectors
w = np.hstack((eig_pairs[0][1].reshape(19,1), eig_pairs[1][1].reshape(19,1),eig_pairs[2][1].reshape(19,1)))

print "w matrix dimensions - ", w.shape
print "input dims - ", inputMat.shape

#Calculate Y with the help of 1st eigen vector
Y = inputMat.dot(w[:,0])
Y

#Plot values of Y for the two classes
fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(1, 1, 1)
ax.set_title('Results from own LDA implementations')
ax.set_xlabel(r'$W_1$')
ax.set_ylabel('')
ax.plot(Y[:20], np.zeros(20), linestyle='None', marker='o', markersize=7, color='red', label='NSCLC')
ax.plot(Y[20:], np.zeros(20), linestyle='None', marker='o', markersize=7, color='blue', label='SCLC')

ax.legend()

fig.show()

#----------------------------------------------------
# Compare LDA implementation with sklearn LDA results
#----------------------------------------------------
X = inputMat
y = np.concatenate((np.zeros(20), np.ones(20)))

II_0 = np.where(y==0)
II_1 = np.where(y==1)

II_0 = II_0[0]
II_1 = II_1[0]


# apply sklearn LDA to cell line data
sklearn_LDA = LDA(n_components=2)
sklearn_LDA_projection = sklearn_LDA.fit_transform(X, y)
#sklearn_LDA_projection = -sklearn_LDA_projection

# plot the projections
fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(1, 1, 1)
ax.set_title('Results from applying sklearn LDA')
ax.set_xlabel(r'$W_1$')
ax.set_ylabel('')
ax.plot(sklearn_LDA_projection[II_0], np.zeros(len(II_0)), linestyle='None', marker='o', markersize=7, color='red', label='NSCLC')
ax.plot(sklearn_LDA_projection[II_1], np.zeros(len(II_1)), linestyle='None', marker='o', markersize=7, color='blue', label='SCLC')
ax.legend()

fig.show()




