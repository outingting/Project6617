import numpy as np
import cvxpy as cp
from scipy.fft import dct


def F(theta):
    if theta.ndim == 1:
        theta = np.expand_dims(theta, 0)
    return -np.sum((theta - 0.5) ** 2, axis=tuple(range(theta.ndim)[1:]))

def get_dct_mtx(d):
    # DCT matrix
    # unitary, symmetric and real
    # Orthonormal eigenbasis for structured H
    n = 2*d
    i_idx = np.array([range(n//2 )])
    idx = 2 * np.transpose(i_idx) @ i_idx
    dct_mtx = np.cos(idx*np.pi / n) * 2 / np.sqrt(d)
    dct_mtx[0,0] = 1
    dct_mtx[0,d-1] = 1
    dct_mtx[d-1,0] = 1
    dct_mtx[d-1,d-1] = (-1)**(d)
    return dct_mtx


def Hessian_LP_structured(sigma, theta, num_samples):
    # LP formulation to estimate Hessian
    # Minimizing over the space of matrices of the form
    # shown in the example 7 & 8 in the reference
    # [MATRICES DIAGONALIZED BY THE DISCRETECOSINE AND DISCRETE SINE TRANSFORMS]
    
    d = len(theta)
    n = num_samples
    
    epsilons = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d), size = n)
    
    # Define and solve the LP for Hessian here
    y = (F(theta + sigma * epsilons) + F(theta - sigma * epsilons) - 2 * F(theta)) / (sigma ** 2)
    
    var_z = cp.Variable(n)
    var_H_diag = cp.Variable(d)
    
    # Lower triangular mtx H
    dct_mtx = get_dct_mtx(d)
    obj = sum(var_z)
    
    constraints = []
    for i in range(n):
        Uv = epsilons[i:i+1,:] @ dct_mtx
        Uv_sq = Uv * Uv
        constraints += [var_z[i] >= y[i] - Uv_sq @ var_H_diag]
        constraints += [var_z[i] >= - y[i] + Uv_sq @ var_H_diag]
    for i in range(d):
        constraints += [var_H_diag[i] <= 0]
        
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.GLPK, eps=1e-6, glpk={'msg_lev': 'GLP_MSG_OFF'})
    
    # 
    if prob.status == 'optimal':
        return dct_mtx @ np.diag(var_H_diag.value) @ np.transpose(dct_mtx)
    
    return None


sigma = 0.01
theta = np.random.uniform(-10,10,3)

n = 50
H_structured = Hessian_LP_structured(sigma, theta, n)

print(H_structured)
