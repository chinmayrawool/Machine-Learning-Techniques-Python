# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 21:25:40 2017

@author: Chinmay Rawool
"""

#Neural Network
import numpy as np
import matplotlib.pyplot as plt
import math


def cost_function(z):
    e = math.e
    m,n = z.shape
    
    a = np.array([[1]], dtype='float')
    for i in range(0,m):
        z[i]
        value = 1/(1+e**z[i])
        print(value)
        
        a = np.concatenate((a,[value]),axis=0)
        
    cost = a
    cost
    
    return cost

def ddx_cost(z):
    return z * (1-z)

def forward_propagation(x,y,theta_1,theta_2):

    one = np.array([[1]],dtype='float')
    #array([[1, 2, 3]]).T
    inputMat = x.transpose()
    inputMat
    
    y = y.transpose()
    
    #Add bias unit to a1
    a1 = np.concatenate((one,inputMat),axis=0)
    a1
    #Input layer: x0,x1,x2
    z2 = np.matmul(theta_1,a1)
    z2
    
    a2 = cost_function(z2)
    a2
    
    z3 = np.matmul(theta_2,a2)
    z3
    
    a3 = cost_function(z3)
    a3
    
    a3 = np.delete(a3,0,0)
    a3
    
    #Total Cost 
    Total = 0.5*((y[0]-a3[0])**2 + (y[1]-a3[1])**2)
    Total
    
    return a1,a2,a3,Total

def back_propagation(a1,a2,a3,theta_1,theta_2,y):
    alpha =0.5
    
    error = a3 - y.T
    #delta 3
    d3 = error * ddx_cost(a3)
    d3
    
    m,n = a2.shape
    a2_row = np.reshape(a2,(n,m))
    a2_row.shape
    new_theta_2 = np.matmul(d3,a2_row) * alpha
    new_theta_2
    
    #(a3-y) * (a3*(1-a3)) * theta_2(:,1) *(a2*(1-a2))
    t21 = theta_2[:,1]
    t21.shape
    t21 = np.reshape(t21,(2,1))
    t21
    
    t22 = theta_2[:,2]
    t22.shape
    t22 = np.reshape(t22,(2,1))
    t22
    
    der_a21 = ddx_cost(a2[1])
    der_a22 = ddx_cost(a2[2])
    
    d21 = error * ddx_cost(a3) * t21 * der_a21
    d22 = error * ddx_cost(a3) * t22 * der_a22
    
    #delta 2
    d2 = np.zeros((2,1),dtype='float')
    d2[0,0] = d21.sum(axis=0)
    d2[1,0] = d22.sum(axis=0)
    d2
    
    m,n = a1.shape
    a1_row = np.reshape(a1,(n,m))
    a1_row.shape
    new_theta_1 = np.matmul(d2,a1_row) * alpha
    new_theta_1
    
    new_theta_1 = theta_1 + new_theta_1
    new_theta_2 = theta_2 + new_theta_2
    
    return new_theta_1,new_theta_2,d3,d2


def main():

    # ----------------------------------------------------------------
    # Q.2. Neural Networks Implementation
    # ----------------------------------------------------------------
    
    no_of_input_nodes = 2
    no_of_hidden_nodes = 2
    no_of_output_nodes = 2
    
    #Initialize the input and the ouput
    x = np.array([[0.05,0.1]],dtype='float')
    y = np.array([[0.01,0.99]],dtype='float')
    
    np.random.seed(10)
    #Size of theta array: (s_j+1,(s_j + 1)) = 2x3
    #Theta 1: size(2,3) of uniformly distributed random values between 0 and 1
    theta_1 = np.random.uniform(low=0.0,high=1.0,size=(no_of_hidden_nodes,no_of_input_nodes+1))
    theta_1
    #Theta 2: size(2,3) of uniformly distributed random values between 0 and 1
    theta_2 = np.random.uniform(low=0.0,high=1.0,size=(no_of_output_nodes,no_of_hidden_nodes+1))
    theta_2
    
    total_cost = np.zeros((40001,2),dtype='float')
    theta1_value = np.zeros((40001,7),dtype='float')
    theta2_value = np.zeros((40001,7),dtype='float')
    
    for i in range(40001):
        #Forward Propagation
        a1,a2,a3,Total = forward_propagation(x,y,theta_1,theta_2)
        #Total Cost per iteration
        total_cost[i,0]=i+1
        total_cost[i,1]=Total
        #Theta 1 parameters 
        theta1_value[i,0]=i+1
        theta1_value[i,1]=theta_1[0,0]
        theta1_value[i,2]=theta_1[0,1]
        theta1_value[i,3]=theta_1[0,2]
        theta1_value[i,4]=theta_1[1,0]
        theta1_value[i,5]=theta_1[1,1]
        theta1_value[i,6]=theta_1[1,2]
        #Theta 2 parameters
        theta2_value[i,0]=i+1
        theta2_value[i,1]=theta_2[0,0]
        theta2_value[i,2]=theta_2[0,1]
        theta2_value[i,3]=theta_2[0,2]
        theta2_value[i,4]=theta_2[1,0]
        theta2_value[i,5]=theta_2[1,1]
        theta2_value[i,6]=theta_2[1,2]
        
        #Backward Propagation
        new_t1, new_t2, delta_3, delta_2 = back_propagation(a1,a2,a3,theta_1,theta_2,y)
        #Update Theta values for next iteration
        theta_1 = new_t1
        theta_2 = new_t2
    
    #Plot
    #Plot total cost vs iterations
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(total_cost[:,0], total_cost[:,1], color='red',s=1, label='Total Cost vs iterations') 
    ax.legend()
    fig.show()      
        
    #Plot theta1 parameters vs iterations
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(2, 3, 1)
    ax.scatter(theta1_value[:,0], theta1_value[:,1], color='black',s=1, label='Theta1_10 vs iterations') 
    ax.legend()
    
    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(theta1_value[:,0], theta1_value[:,2], color='black',s=1, label='Theta1_11 vs iterations') 
    ax.legend()
    
    ax = fig.add_subplot(2, 3, 3)
    ax.scatter(theta1_value[:,0], theta1_value[:,3], color='black',s=1, label='Theta1_12 vs iterations') 
    ax.legend()
    
    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(theta1_value[:,0], theta1_value[:,4], color='black',s=1, label='Theta1_20 vs iterations') 
    ax.legend()
    ax = fig.add_subplot(2, 3, 5)
    ax.scatter(theta1_value[:,0], theta1_value[:,5], color='black',s=1, label='Theta1_21 vs iterations') 
    ax.legend()
    ax = fig.add_subplot(2, 3, 6)
    ax.scatter(theta1_value[:,0], theta1_value[:,6], color='black',s=1, label='Theta1_22 vs iterations') 
    ax.legend()
    fig.show()      
        
    
    #Plot Theta2 parameters vs iterations    
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(2, 3, 1)
    ax.scatter(theta2_value[:,0], theta2_value[:,1], color='black',s=1, label='Theta2_10 vs iterations') 
    ax.legend()
    
    ax = fig.add_subplot(2, 3, 2)
    ax.scatter(theta2_value[:,0], theta2_value[:,2], color='black',s=1, label='Theta2_11 vs iterations') 
    ax.legend()
    
    ax = fig.add_subplot(2, 3, 3)
    ax.scatter(theta2_value[:,0], theta2_value[:,3], color='black',s=1, label='Theta2_12 vs iterations') 
    ax.legend()
    
    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(theta2_value[:,0], theta2_value[:,4], color='black',s=1, label='Theta2_20 vs iterations') 
    ax.legend()
    ax = fig.add_subplot(2, 3, 5)
    ax.scatter(theta2_value[:,0], theta2_value[:,5], color='black',s=1, label='Theta2_21 vs iterations') 
    ax.legend()
    ax = fig.add_subplot(2, 3, 6)
    ax.scatter(theta2_value[:,0], theta2_value[:,6], color='black',s=1, label='Theta2_22 vs iterations') 
    ax.legend()
    fig.show() 

    
if __name__ == '__main__':
    main()    
    
    
    
    