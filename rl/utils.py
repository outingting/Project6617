import numpy as np
import cvxpy as cp
from sklearn.linear_model import LinearRegression
from scipy.stats import ortho_group

def Gradient_LP(y, epsilons):
    """
    y = (F(theta + sigma*epsilons) - F(theta)) / sigma
    epsilons: the perturbations with UNIT VARIANCE
    """
    _, d = epsilons.shape
    var_g = cp.Variable(d)
    constraints = []
    obj = cp.norm1(y-epsilons @ var_g)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.GUROBI)
    if prob.status == 'optimal':
        return var_g.value
    raise ValueError("Gradient LP did not converge: %s" % prob.status)

def Gradient_L2(y, epsilons):
    """
    y = (F(theta + sigma*epsilons) - F(theta)) / sigma
    epsilons: the perturbations with UNIT VARIANCE
    """
    reg = LinearRegression(fit_intercept=False)
    reg.fit(epsilons, y)
    return reg.coef_


def Hessian_LP(y, epsilons):
    """
    y = (F(theta + sigma*epsilons) + F(theta - sigma*epsilons) - 2*F(theta)) / (sigma**2)
    epsilons: the perturbations in the above expression (note that it does not
    include the sigma)
    """
    n, d = epsilons.shape
    
    X = np.zeros((n, d*(d+1)//2))
    idx = 0
    for j in range(d):
        X[:,idx] = epsilons[:,j]**2
        idx += 1
        if j == d-1:
            break
        X[:,idx:idx+d-j-1] = 2 * epsilons[:,j:j+1] * epsilons[:,j+1:]
        idx += d-j-1
    
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
        idx = 0
        for j in range(d):
            H[j,j:] = var_H[idx:idx+d-j].value
            H[j:,j] = var_H[idx:idx+d-j].value
            idx += d-j
        return H
    raise ValueError("Hessian LP did not converge: %s" % prob.status)


def get_dct_mtx(d):
    """
    DCT matrix
    unitary, symmetric and real
    Orthonormal eigenbasis for structured H
    """
    n = 2*(d-1)
    dct_mtx = np.zeros([d,d])
    i_idx = np.array([range(n//2 )])
    i_idx = i_idx[:,1:]
    idx = 2 * np.transpose(i_idx) @ i_idx
    dct_mtx[1:-1,1:-1] = np.cos(idx*np.pi / n) * 2
    for ii in range(d):
        dct_mtx[ii,0] = np.sqrt(2)
        dct_mtx[0,ii] = np.sqrt(2)
        dct_mtx[ii,-1] = (-1)**(ii) * np.sqrt(2)
        dct_mtx[-1,ii] = (-1)**(ii) * np.sqrt(2)
    dct_mtx[0, 0] = 1
    dct_mtx[0, d - 1] = 1
    dct_mtx[d - 1, 0] = 1
    dct_mtx[d - 1, d - 1] = (-1)**(d-1)
    dct_mtx = dct_mtx / np.sqrt(d)
    return dct_mtx

def get_PTinverse(diag_H, PT_threshold=-1):
    """
    Obtain the PT inverse (Negative definite version)
    Input: 
        - diag_H, an array represeting the diagonal of the diagonal matrix
        obtained from svd [diag(\Lambda), H = U \Lambda V]
    Output: 
        - diagonal of the diagonal matrix of the svd of PT inverse
        [diag(\Lambda^{-1}_{PT})]
    """
    # Assume we are solving a maximization problem and the estimated Hessian is expected to be Negative definite
    diag_H[diag_H >= PT_threshold] = PT_threshold
    return diag_H ** (-1)

def Hessian_L2_structured(y, epsilons):
    reg = LinearRegression(fit_intercept=False)

    _, d = epsilons.shape
    dct_mtx = get_dct_mtx(d)
    Uv_sq = (epsilons@dct_mtx)**2

    reg.fit(Uv_sq, y)
    return reg.coef_, dct_mtx

def Hessian_LP_structured(y, epsilons):
    """
    y = (F(theta + sigma * epsilons) + F(theta - sigma * epsilons) - 2 * F(theta)) / (sigma ** 2)
    """
    # LP formulation to estimate Hessian
    # Minimizing over the space of matrices of the form
    # shown in the example 7 & 8 in the reference
    # [MATRICES DIAGONALIZED BY THE DISCRETECOSINE AND DISCRETE SINE TRANSFORMS]
    
    n, d = epsilons.shape
    
    var_H_diag = cp.Variable(d)
    dct_mtx = get_dct_mtx(d)

    Uv_sq = cp.square(epsilons@dct_mtx)
    yhat = Uv_sq @ var_H_diag
    constraints = [var_H_diag <= 0]
    prob = cp.Problem(cp.Minimize(cp.norm(yhat-y, 1)), constraints)
    prob.solve(solver=cp.GUROBI)
    if prob.status == 'optimal':
        # H = dct_mtx @ np.diag(var_H_diag.value) @ np.transpose(dct_mtx)
        return var_H_diag.value, dct_mtx
    raise ValueError("LP not optimized for the structured Hessian method. Problem Status: %s" % prob.status) 


def orthogonal_gaussian(d, n_samples):
    blocks = n_samples//d + (n_samples%d >0)
    out = np.concatenate([ortho_group.rvs(d) for _ in range(blocks)])[:n_samples]
    norms = np.sqrt(np.random.chisquare(d, size=n_samples))
    out = (out.T * norms).T
    return out

