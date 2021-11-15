#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:36:32 2021

@author: Xujia
"""


import numpy as np
import cvxpy as cp


def F(theta):
    if theta.ndim == 1:
        theta = np.expand_dims(theta, 0)
    return -np.sum((theta - 0.5) ** 2, axis=tuple(range(theta.ndim)[1:]))


def Hessian_LP(sigma, theta, num_samples):
    
    d = len(theta)
    n = num_samples
    
    epsilons = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d), size = n)
    
    y = (F(theta + sigma*epsilons) + F(theta - sigma*epsilons) - 2*F(theta)) / (sigma**2)
    X = np.zeros((n, d*(d+1)//2))
    for j in range(d):
        X[:,j*(j+1)//2:(j+1)*(j+2)//2-1] = 2 * epsilons[:,j:j+1] * epsilons[:,:j]
        X[:,(j+1)*(j+2)//2-1] = epsilons[:,j]**2
    
    var_z = cp.Variable(n)
    var_H = cp.Variable(d*(d+1)//2)
    
    obj = sum(var_z)
    
    constraints = []
    for i in range(n):
        constraints += [var_z[i] >= y[i] - X[i] @ var_H]
        constraints += [var_z[i] >= - y[i] + X[i] @ var_H]
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.GLPK, eps=1e-6, glpk={'msg_lev': 'GLP_MSG_OFF'})

    if prob.status == 'optimal':
        H = np.zeros((d,d))
        for j in range(d):
            H[j,0:j+1] = var_H[j*(j+1)//2:(j+1)*(j+2)//2].value
            H[1:j+1,j] = H[j,1:j+1]
    
    return H


sigma = 0.01
theta = np.random.uniform(-10,10,5)
n = 50

H = Hessian_LP(sigma, theta, n)
print(H)













