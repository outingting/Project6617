# Different methods to optimize

# ES_vanilla_gradient is Algorithm 1 in https://arxiv.org/abs/1703.03864
# ES_Hessian is from the project proposal, and we use the same update rule as that in HessAware
# Hess_Aware is Algorithm "HessAware" in https://arxiv.org/abs/1812.11377
# LP_Hessian is to use LP to solve for estimates of gradient & Hessian, and do a Newton's update
# LP_Hessian_structured is a modified version of LP_Hessian, while we are minimizing over the space of matrices of a specific form

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import cvxpy as cp
import copy
from scipy.linalg import cholesky
from numpy.random import standard_normal
from numpy.linalg import LinAlgError

#########################################################################################################

# Helper functions of the optimizers

def Adam(dx, m, v, learning_rate, t, eps = 1e-8, beta1 = 0.9, beta2 = 0.999):
    m = beta1 * m + (1 - beta1) * dx
    mt = m / (1 - beta1 ** t)
    v = beta2 * v + (1-beta2) * (dx **2)
    vt = v / (1 - beta2 ** t)
    update = learning_rate * mt / (np.sqrt(vt) + eps)
    return(update, m, v)


def LP_Gradient(y, epsilons):
    """
    y = (F(theta_t + sigma*epsilons) - F(theta_t)) / sigma
    epsilons: the perturbations with UNIT VARIANCE
    """
    n, d = epsilons.shape
    var_z = cp.Variable(n)
    var_g = cp.Variable(d)
    obj = sum(var_z)
    constraints = [var_z >= y - epsilons @ var_g, var_z >= -y + epsilons @ var_g]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.GLPK, eps=1e-6, glpk={'msg_lev': 'GLP_MSG_OFF'})
    if prob.status == 'optimal':
        return var_g.value
    return None

def get_dct_mtx(d):
    # DCT matrix
    # unitary, symmetric and real
    # Orthonormal eigenbasis for structured H
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

# Obtain the PT inverse (Negative definite version)
# Input: diagonal of the diagonal matrix obtained from svd [diag(\Lambda), H = U \Lambda V]
# Output: diagonal of the diagonal matrix of the svd of PT inverse [diag(\Lambda^{-1}_{PT})]
def get_PTinverse(diag_H, PT_threshold=-1e0):
    # Assume we are solving a maximization problem and the estimated Hessian is expected to be Negative definite
    diag_H[diag_H >= PT_threshold] = PT_threshold
    return diag_H ** (-1)

def simulation_Gradient(F_val, num_samples, epsilons, sigma):
    F_val = F_val.reshape(1, num_samples)
    g = (F_val @ epsilons).ravel() / (sigma * num_samples)
    return g

#########################################################################################################

def ES_vanilla_gradient(F, alpha, sigma, theta_0, num_samples, time_steps, seed=1):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        #**** sample epsilons ****#
        epsilons = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d), size = num_samples) # n by d
        #**** compute function values ****#
        F_val = F(theta_t + sigma * epsilons)
        count += num_samples
        #**** update theta ****#
        new_theta = theta_t
        F_val = F_val.reshape(1, num_samples)
        update = (F_val @ epsilons).ravel()
        new_theta += alpha / (num_samples*sigma) * update
        theta_t = new_theta
        #**** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))
    return theta_t, F(theta_t), None, lst_evals, lst_f

#########################################################################################################

def ES_Hessian(F, alpha, sigma, theta_0, num_samples, time_steps, p = 1, H_lambda = 0, seed=1):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = [] 
    d = theta_0.shape[0]
    theta_t = copy.deepcopy(theta_0)
    n = num_samples
    H = None
    for t in range(time_steps):
        
        #**** sample epsilons ****#
        epsilons = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d), size = num_samples) # n by d

        #**** compute Hessian every p steps ****#
        if t % p == 0:
            eps = np.expand_dims(epsilons, -1)
            F_plus = F(theta_t + sigma*epsilons)
            count += num_samples
            H_samples = ((eps*np.transpose(eps, (0, 2, 1)) - np.identity(d))* F_plus.reshape(-1, 1, 1))/(sigma**2)
            H = H_samples.mean(axis=0)
            u, s, vh = np.linalg.svd(H)
            H_nh = u @ np.diag(s**-0.5) @ vh
            H_nh_3d = np.ones((n, d, d)) * H_nh
            
        #**** update theta: compute g ****#
        Fs = F(theta_t +  sigma* np.transpose((H_nh @ np.transpose(epsilons) )) ) - F(theta_t)
        count += num_samples
        eps = np.expand_dims(epsilons, -1)
        g_samples =  H_nh_3d @ eps * Fs.reshape(-1, 1, 1)/sigma
        g = g_samples.mean(axis=0).ravel()
        
        #**** update theta: the rest ****#
        new_theta = theta_t + alpha * g
        theta_t = new_theta

        #**** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))
        
    return theta_t, F(theta_t), H, lst_evals, lst_f

#########################################################################################################

def ES_Hessian_v2(F, alpha, sigma, theta_0, num_samples, time_steps, p = 1, H_lambda = 0, seed=1):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = [] 
    d = theta_0.shape[0]
    theta_t = copy.deepcopy(theta_0)
    n = num_samples
    H = None
    for t in range(time_steps):
        
        #**** sample epsilons ****#
        epsilons = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d), size = num_samples) # n by d

        #**** compute Hessian every p steps ****#
        if t % p == 0:
            eps = np.expand_dims(epsilons, -1)
            F_plus = F(theta_t + sigma*epsilons)
            F_minus = F(theta_t - sigma*epsilons)
            count += 2*num_samples
            H_samples = ((eps*np.transpose(eps, (0, 2, 1)) - np.identity(d))* F_plus.reshape(-1, 1, 1))/(sigma**2)
            H = H_samples.mean(axis=0)

        #**** compute g by simulation ****#
        g = simulation_Gradient(np.concatenate([F_plus, F_minus]), 2*num_samples, np.vstack([epsilons, -epsilons]), sigma)
            
        #**** update using Newton's method ****#
        theta_t -= alpha * np.linalg.inv(H)@g

        #**** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))
        
    return theta_t, F(theta_t), H, lst_evals, lst_f

#########################################################################################################

def Hess_Aware(F, alpha, sigma, theta_0, num_samples, time_steps, p = 1, H_lambda = 0, seed=1):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    H = None
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        #**** sample epsilons ****#
        epsilons = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d), size = num_samples) # n by d

        #**** compute function values ****#
        F_plus = F(theta_t + sigma * epsilons)
        F_minus = F(theta_t - sigma * epsilons)
        F_val = np.array([F(theta_t)] * num_samples).ravel()
        count += 2*num_samples

        if t % p == 0:
            H = np.zeros((d,d))
            eps = np.expand_dims(epsilons, -1)
            eet = eps*np.transpose(eps, (0, 2, 1))
            H_samples = (F_plus.reshape(-1,1,1) + F_minus.reshape(-1,1,1) - 2*F_val.reshape(-1,1,1) ) * eet
            H = H_samples.mean(axis=0) / (2*sigma**2)
            u, s, vh = np.linalg.svd(H)
            H_nh = u @ np.diag(s**-0.5) @ vh
            H_nh_3d = np.ones((num_samples, d, d)) * H_nh

        #**** update theta: compute g ****#
        Fs = F(theta_t +  sigma* np.transpose((H_nh @ np.transpose(epsilons) )) ) - F(theta_t)
        count += num_samples
        eps = np.expand_dims(epsilons, -1)
        g_samples =  H_nh_3d @ eps * Fs.reshape(-1, 1, 1)/sigma
        g = g_samples.mean(axis=0).ravel()
        
        #**** update theta: the rest ****#
        new_theta = theta_t + alpha * g
        theta_t = new_theta

        #**** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))

    return theta_t, F(theta_t), H, lst_evals, lst_f

#########################################################################################################

def LP_Hessian(F, alpha, sigma, theta_0, num_samples, time_steps, seed=1):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    n = num_samples
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        #**** sample epsilons, record some parameters & function values ****#
        epsilons = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d), size = num_samples) # n by d
        # n, d = epsilons.shape
        F_plus = F(theta_t + sigma*epsilons)
        F_minus = F(theta_t - sigma*epsilons)
        count += 2*num_samples

        #**** estimate Hessian using the LP method ****#
        y = (F_plus + F_minus - 2*F(theta_t)) / (sigma**2)
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
        else:
            print("LP not optimized for LP Hessian method.")
            return None

        #**** update using Newton's method ****#
        theta_t -= alpha * np.linalg.inv(H)@g

        #**** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))

    return theta_t, F(theta_t), H, lst_evals, lst_f

#########################################################################################################

def LP_Hessian_structured(F, alpha, sigma, theta_0, num_samples, time_steps, H_lambda = 1e-6, seed=1):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    n = num_samples
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        #**** sample epsilons, record some parameters & function values ****#
        epsilons = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d), size = num_samples) # n by d
        F_plus = F(theta_t + sigma*epsilons)
        F_minus = F(theta_t - sigma*epsilons)
        count += 2*num_samples

        #**** estimate Hessian ****#
        y = (F_plus + F_minus - 2 * F(theta_t))/ (sigma ** 2)
        var_z = cp.Variable(n)
        var_H_diag = cp.Variable(d)
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
        if prob.status == 'optimal':
            H = dct_mtx @ np.diag(var_H_diag.value) @ np.transpose(dct_mtx)
            #if np.linalg.det(H) == 0:
            H -= H_lambda * np.identity(d)
        else:
            print("LP not optimized for the structured Hessian method.")
            return None

        #**** estimate gradient (by using the method above) ****#
        y = (F_plus - F(theta_t)) / sigma
        g = LP_Gradient(y, epsilons)

        #**** update using Newton's method ****#
        theta_t -= alpha * np.linalg.inv(H)@g

        #**** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))

    return theta_t, F(theta_t), H, lst_evals, lst_f

#########################################################################################################

# PT inverse
# Non- antithetic
# fixed step size
# LP gradient
## [not stable]
def LP_Hessian_structured_v2(F, alpha, sigma, theta_0, num_samples, time_steps, PT_threshold=-1e0, seed=1):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    n = num_samples
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        # **** sample epsilons, record some parameters & function values ****#
        epsilons = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=num_samples)  # n by d
        F_plus = F(theta_t + sigma * epsilons)
        F_minus = F(theta_t - sigma * epsilons)
        count += 2 * num_samples

        # **** estimate Hessian ****#
        y = (F_plus + F_minus - 2 * F(theta_t)) / (sigma ** 2)
        var_z = cp.Variable(n)
        var_H_diag = cp.Variable(d)
        dct_mtx = get_dct_mtx(d)
        obj = sum(var_z)
        constraints = []
        for i in range(n):
            Uv = epsilons[i:i + 1, :] @ dct_mtx
            Uv_sq = Uv * Uv
            constraints += [var_z[i] >= y[i] - Uv_sq @ var_H_diag]
            constraints += [var_z[i] >= - y[i] + Uv_sq @ var_H_diag]
        for i in range(d):
            constraints += [var_H_diag[i] <= 0]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.GLPK, eps=1e-6, glpk={'msg_lev': 'GLP_MSG_OFF'})

        if not prob.status == 'optimal':
            print("LP not optimized for the structured Hessian method.")
            return None

        # **** estimate gradient (by using the method above) ****#
        y = (F_plus - F(theta_t)) / sigma
        g = LP_Gradient(y, epsilons)

        # **** update using Newton's method ****#
        # import pdb; pdb.set_trace()
        theta_t -= alpha * dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value,PT_threshold)) @ (dct_mtx @ g))

        # Can delete, not really useful
        H = dct_mtx @ np.diag(var_H_diag.value) @ dct_mtx

        # **** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))

    return theta_t, F(theta_t), H, lst_evals, lst_f

#########################################################################################################

# PT inverse
# antithetic
# fixed step size
# LP gradient
## [not stable]
def LP_Hessian_structured_v3(F, alpha, sigma, theta_0, num_samples, time_steps, PT_threshold=-1e0, seed=1):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    n = num_samples
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        # **** sample epsilons, record some parameters & function values ****#
        epsilons = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=num_samples)  # n by d
        epsilons_antithetic = np.vstack([epsilons, -epsilons])
        F_plus = F(theta_t + sigma * epsilons)
        F_minus = F(theta_t - sigma * epsilons)
        F_theta = F(theta_t)
        count += 2 * num_samples

        # **** estimate Hessian ****#
        y = (F_plus + F_minus - 2 * F_theta) / (sigma ** 2)
        var_z = cp.Variable(n)
        var_H_diag = cp.Variable(d)
        dct_mtx = get_dct_mtx(d)
        obj = sum(var_z)
        constraints = []
        for i in range(n):
            Uv = epsilons[i:i + 1, :] @ dct_mtx
            Uv_sq = Uv * Uv
            constraints += [var_z[i] >= y[i] - Uv_sq @ var_H_diag]
            constraints += [var_z[i] >= - y[i] + Uv_sq @ var_H_diag]
        for i in range(d):
            constraints += [var_H_diag[i] <= 0]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.GLPK, eps=1e-6, glpk={'msg_lev': 'GLP_MSG_OFF'})
        if not prob.status == 'optimal':
            print("LP not optimized for the structured Hessian method.")
            return None

        # **** estimate gradient (by using the method above) ****#
        y_antithetic = (np.concatenate([F_plus,F_minus]) - F_theta) / sigma
        g = LP_Gradient(y_antithetic, epsilons_antithetic)
        #####

        # **** update using Newton's method ****#
        theta_t -= alpha * dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value,PT_threshold)) @ (dct_mtx @ g))

        # Can delete, not really useful
        H = dct_mtx @ np.diag(var_H_diag.value) @ dct_mtx

        # import pdb; pdb.set_trace()
        # **** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))

    return theta_t, F(theta_t), H, lst_evals, lst_f

#########################################################################################################

# PT inverse
# antithetic
# adaptive step size (backtracking)
# LP gradient
def LP_Hessian_structured_v4(F, alpha, sigma, theta_0, num_samples, time_steps, PT_threshold=-1e0, seed=1, beta=0.5):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    n = num_samples
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        eta = 1
        # **** sample epsilons, record some parameters & function values ****#
        epsilons = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=num_samples)  # n by d
        epsilons_antithetic = np.vstack([epsilons, -epsilons])
        F_plus = F(theta_t + sigma * epsilons)
        F_minus = F(theta_t - sigma * epsilons)
        F_theta = F(theta_t)
        count += 2 * num_samples + 1

        # **** estimate Hessian ****#
        y = (F_plus + F_minus - 2 * F_theta) / (sigma ** 2)
        var_z = cp.Variable(n)
        var_H_diag = cp.Variable(d)
        dct_mtx = get_dct_mtx(d)
        obj = sum(var_z)
        constraints = []
        for i in range(n):
            Uv = epsilons[i:i + 1, :] @ dct_mtx
            Uv_sq = Uv * Uv
            constraints += [var_z[i] >= y[i] - Uv_sq @ var_H_diag]
            constraints += [var_z[i] >= - y[i] + Uv_sq @ var_H_diag]
        for i in range(d):
            constraints += [var_H_diag[i] <= 0]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.GLPK, eps=1e-6, glpk={'msg_lev': 'GLP_MSG_OFF'})
        if not prob.status == 'optimal':
            print("LP not optimized for the structured Hessian method:(")
            return None

        # **** estimate gradient (by using the method above) ****#
        y_antithetic = (np.concatenate([F_plus,F_minus]) - F_theta) / sigma
        g = LP_Gradient(y_antithetic, epsilons_antithetic)

        # **** update using Newton's method ****#
        theta_t_ = theta_t -  eta * dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value, PT_threshold)) @ (dct_mtx @ g))
        F_t = F(theta_t)

        # backtracking
        cnt = 0
        while F(theta_t_)[0] < (F_t + alpha * eta * np.transpose(g) @ dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value, PT_threshold)) @ (dct_mtx @ g)))[0]:
            if cnt >= 30:
                break
            cnt += 1
            eta *= beta
            theta_t_ = theta_t - eta * dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value, PT_threshold)) @ (dct_mtx @ g))
        theta_t = theta_t_
        count += cnt

        # Can delete, not really useful
        H = dct_mtx @ np.diag(var_H_diag.value) @ dct_mtx

        # import pdb; pdb.set_trace()
        # **** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))

    return theta_t, F(theta_t), H, lst_evals, lst_f

#########################################################################################################

# PT inverse
# antithetic
# adaptive step size (backtracking)
# simulation gradient
def LP_Hessian_structured_v5(F, alpha, sigma, sigma_g, theta_0, num_samples, time_steps, PT_threshold=-1e0, seed=1, beta=0.5):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    n = num_samples
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        eta = 1
        # **** sample epsilons, record some parameters & function values ****#
        epsilons = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=num_samples)  # n by d
        epsilons_antithetic = np.vstack([epsilons, -epsilons])
        F_plus = F(theta_t + sigma * epsilons)
        F_minus = F(theta_t - sigma * epsilons)
        F_theta = F(theta_t)
        count += 2 * num_samples + 1

        # **** estimate Hessian ****#
        y = (F_plus + F_minus - 2 * F_theta) / (sigma ** 2)
        var_z = cp.Variable(n)
        var_H_diag = cp.Variable(d)
        dct_mtx = get_dct_mtx(d)
        obj = sum(var_z)
        constraints = []
        for i in range(n):
            Uv = epsilons[i:i + 1, :] @ dct_mtx
            Uv_sq = Uv * Uv
            constraints += [var_z[i] >= y[i] - Uv_sq @ var_H_diag]
            constraints += [var_z[i] >= - y[i] + Uv_sq @ var_H_diag]
        for i in range(d):
            constraints += [var_H_diag[i] <= 0]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.GLPK, eps=1e-6, glpk={'msg_lev': 'GLP_MSG_OFF'})
        if not prob.status == 'optimal':
            print("LP not optimized for the structured Hessian method:(")
            return None

        # **** estimate gradient (by using the method above) ****#
        # y_antithetic = (np.concatenate([F_plus,F_minus]) - F_theta) / sigma
        # g = LP_Gradient(y_antithetic, epsilons_antithetic)
        g = simulation_Gradient(np.concatenate([F_plus, F_minus]), 2*num_samples, np.vstack([epsilons, -epsilons]), sigma_g)

        # **** update using Newton's method ****#
        theta_t_ = theta_t - eta * dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value, PT_threshold)) @ (dct_mtx @ g))
        F_t = F(theta_t)

        # backtracking
        cnt = 0
        while F(theta_t_)[0] < (F_t + alpha * eta * np.transpose(g) @ dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value, PT_threshold)) @ (dct_mtx @ g)))[0]:
            if cnt >= 30:
                break
            cnt += 1
            eta *= beta
            theta_t_ = theta_t - eta * dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value, PT_threshold)) @ (dct_mtx @ g))
        theta_t = theta_t_
        count += cnt

        # Can delete, not really useful
        H = dct_mtx @ np.diag(var_H_diag.value) @ dct_mtx

        # import pdb; pdb.set_trace()
        # **** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))

    return theta_t, F(theta_t), H, lst_evals, lst_f

#########################################################################################################

# PT inverse
# antithetic
# adaptive step size (backtracking)
# linear regression Hessian (non-positive constrained QP)
def LP_Hessian_structured_v6(F, alpha, sigma, sigma_g, theta_0, num_samples, time_steps, PT_threshold=-1e0, seed=1, beta=0.5):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    n = num_samples
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        eta = 1
        # **** sample epsilons, record some parameters & function values ****#
        epsilons = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=num_samples)  # n by d
        epsilons_antithetic = np.vstack([epsilons, -epsilons])
        F_plus = F(theta_t + sigma * epsilons)
        F_minus = F(theta_t - sigma * epsilons)
        F_theta = F(theta_t)
        count += 2 * num_samples + 1

        # Estimate Hessian
        y = (F_plus + F_minus - 2*F_theta) / (sigma**2)
        var_H_diag = cp.Variable(d)
        dct_mtx = get_dct_mtx(d)
        constraints = []
        obj = 0
        for i in range(d):
            constraints += [var_H_diag[i] <= 0]
        for i in range(n):
            Uv = epsilons[i:i + 1, :] @ dct_mtx
            Uv_sq = Uv * Uv
            obj += (y[i] - Uv_sq @ var_H_diag)**2
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.OSQP)
        if not prob.status=='optimal':
            print("QP not optimized for the structured Hessian method:<")
            return None

        # **** estimate gradient (by using the method above) ****#
        y_antithetic = (np.concatenate([F_plus,F_minus]) - F_theta) / sigma
        g = LP_Gradient(y_antithetic, epsilons_antithetic)

        # **** update using Newton's method ****#
        theta_t_ = theta_t - eta * dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value, PT_threshold)) @ (dct_mtx @ g))
        F_t = F(theta_t)

        # backtracking
        cnt = 0
        while F(theta_t_)[0] < (
                F_t + alpha * eta * np.transpose(g) @ dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value, PT_threshold)) @ (dct_mtx @ g)))[
            0]:
            if cnt >= 30:
                break
            cnt += 1
            eta *= beta
            theta_t_ = theta_t - eta * dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value, PT_threshold)) @ (dct_mtx @ g))
        theta_t = theta_t_
        count += cnt

        # Can delete, not really useful
        H = dct_mtx @ np.diag(var_H_diag.value) @ dct_mtx

        # import pdb; pdb.set_trace()
        # **** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))

    return theta_t, F(theta_t), H, lst_evals, lst_f

#########################################################################################################

# PT inverse
# antithetic
# adaptive step size (backtracking)
# linear regression Hessian (unconstrained)
def LP_Hessian_structured_v7(F, alpha, sigma, sigma_g, theta_0, num_samples, time_steps, PT_threshold=-1e0, seed=1, beta=0.7):
    np.random.seed(seed)
    count = 0
    lst_evals = []
    lst_f = []
    d = theta_0.shape[0]
    n = num_samples
    theta_t = copy.deepcopy(theta_0)
    for t in range(time_steps):
        eta = 1
        # **** sample epsilons, record some parameters & function values ****#
        epsilons = np.random.multivariate_normal(mean=np.zeros(d), cov=np.identity(d), size=num_samples)  # n by d
        epsilons_antithetic = np.vstack([epsilons, -epsilons])
        F_plus = F(theta_t + sigma * epsilons)
        F_minus = F(theta_t - sigma * epsilons)
        F_theta = F(theta_t)
        count += 2 * num_samples + 1

        # Estimate Hessian
        y = (F_plus + F_minus - 2*F_theta) / (sigma**2)
        dct_mtx = get_dct_mtx(d)
        V = epsilons @ dct_mtx
        V = V * V
        H_diag = np.linalg.inv(np.transpose(V) @ V + 1e-6 * np.eye(d)) @ (np.transpose(V) @ y)

        # **** estimate gradient (by using the method above) ****#
        y_antithetic = (np.concatenate([F_plus,F_minus]) - F_theta) / sigma
        g = LP_Gradient(y_antithetic, epsilons_antithetic)

        # **** update using Newton's method ****#
        theta_t_ = theta_t - eta * dct_mtx @ (np.diag(get_PTinverse(H_diag, PT_threshold)) @ (dct_mtx @ g))
        F_t = F(theta_t)

        # backtracking
        cnt = 0
        while F(theta_t_)[0] < (
                F_t + alpha * eta * np.transpose(g) @ dct_mtx @ (np.diag(get_PTinverse(H_diag, PT_threshold)) @ (dct_mtx @ g)))[
            0]:
            if cnt >= 30:
                break
            cnt += 1
            eta *= beta
            theta_t_ = theta_t - eta * dct_mtx @ (np.diag(get_PTinverse(H_diag, PT_threshold)) @ (dct_mtx @ g))
        theta_t = theta_t_
        count += cnt

        # Can delete, not really useful
        H = dct_mtx @ np.diag(H_diag) @ dct_mtx

        # **** record current status ****#
        lst_evals.append(count)
        lst_f.append(F(theta_t))

    return theta_t, F(theta_t), H, lst_evals, lst_f


#########################################################################################################

def asebo_function_eval(F, theta, A, n_samples):
    # A = sigma * epsilon, A shape is (n, d)
    # A[i] = sigma * the i-th epsilon
    all_rollouts = np.zeros([n_samples, 2]) # First Row = F_plus, Second Row = F_minus
    for i in range(n_samples):
        all_rollouts[i, 0] = F(theta + A[i])# F plus 
        all_rollouts[i, 1] = F(theta - A[i]) # F minus
    all_rollouts = (all_rollouts - np.mean(all_rollouts)) / (np.std(all_rollouts)  + 1e-8)

    F_diff = np.array(all_rollouts[:, 0] - all_rollouts[:, 1])
    return F_diff


def asebo(F, sigma, learning_rate, decay, k, theta_0, min_samples, num_sensings, max_iter, use_log, threshold):

    theta_t = copy.deepcopy(theta_0)

    if use_log:
        num_sensings = 4 + int(3 * np.log(d))

    m = 0 ; v = 0; d = len(theta_0); 

    k = min(k, d); count = 0; alpha = 1

    n_iter = 1; G = []

    lst_evals = []; lst_f = []

    while n_iter < max_iter:

        # ES method of ASEBO
        if n_iter >= k:
            pca = PCA()
            pca_fit = pca.fit(G)
            var_exp = pca_fit.explained_variance_ratio_
            var_exp = np.cumsum(var_exp)
            n_samples = np.argmax(var_exp > threshold) + 1
            n_samples = max(n_samples, min_samples)
            U = pca_fit.components_[:n_samples]
            UUT = np.matmul(np.transpose(U), U)
            U_ort = pca_fit.components_[n_samples:]
            UUT_ort = np.matmul(np.transpose(U_ort), U_ort)
            if n_iter == k:
                n_samples = num_sensings
        else:
            UUT = np.zeros([d,d])
            alpha = 1
            n_samples = num_sensings

        # get the matrix A (which is the sigma * epsilons)
        np.random.seed(1)
        cov = (alpha/d) * np.eye(d) + ((1-alpha) / n_samples) * UUT
        cov *= sigma
        mu = np.repeat(0, d)
        A = np.zeros((n_samples, d))
        try:
            l = cholesky(cov, check_finite=False, overwrite_a=True)
            for i in range(n_samples):
                try:
                    A[i] = np.zeros(d) + l.dot(standard_normal(d))
                except LinAlgError:
                    A[i] = np.random.randn(d)
        except LinAlgError:
            for i in range(n_samples):
                A[i] = np.random.randn(d)  
        A /= np.linalg.norm(A, axis =-1)[:, np.newaxis]

        # do function evaluations 
        F_diff = asebo_function_eval(F, theta_t, A, n_samples)
        count += 2 * n_samples

        # process the gradient
        g = np.zeros(d)
        for i in range(n_samples):
            eps = A[i, :]
            g += eps *  F_diff[i]
        g /= (2 * sigma)
        if n_iter >= k:
            alpha = np.linalg.norm(np.dot(g, UUT_ort))/np.linalg.norm(np.dot(g, UUT))
        if n_iter == 1:
            G = np.array(g)
        else:
            G *= decay
            G = np.vstack([G, g])
        g/= (np.linalg.norm(g) / d + 1e-8)

        # update the current theta value
        update, m, v = Adam(g, m, v, learning_rate, n_iter)
        theta_t += update
        n_iter += 1
        lst_evals.append(count)
        lst_f.append(F(theta_t))

    return theta_t, F(theta_t), None, lst_evals, lst_f




