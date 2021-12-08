# Different methods to optimize

# ES_vanilla_gradient is Algorithm 1 in https://arxiv.org/abs/1703.03864
# ES_Hessian is from the project proposal, and we use the same update rule as that in HessAware
# Hess_Aware is Algorithm "HessAware" in https://arxiv.org/abs/1812.11377
# LP_Hessian is to use LP to solve for estimates of gradient & Hessian, and do a Newton's update
# LP_Hessian_structured is a modified version of LP_Hessian, while we are minimizing over the space of matrices of a specific form

import numpy as np
from scipy.stats import multivariate_normal
import cvxpy as cp
import copy

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
    n = 2*d
    i_idx = np.array([range(n//2 )])
    idx = 2 * np.transpose(i_idx) @ i_idx
    dct_mtx = np.cos(idx*np.pi / n) * 2 / np.sqrt(d)
    dct_mtx[0,0] = 1
    dct_mtx[0,d-1] = 1
    dct_mtx[d-1,0] = 1
    dct_mtx[d-1,d-1] = (-1)**(d)
    return dct_mtx


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
        else:
            print("LP not optimized for LP Hessian method.")
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

# Obtain the PT inverse (Negative definite version)
# Input: diagonal of the diagonal matrix obtained from svd [diag(\Lambda), H = U \Lambda V]
# Output: diagonal of the diagonal matrix of the svd of PT inverse [diag(\Lambda^{-1}_{PT})]
def get_PTinverse(diag_H, PT_threshold=-0.01):
    # Assume we are solving a maximization problem and the estimated Hessian is expected to be Negative definite
    diag_H[diag_H >= PT_threshold] = PT_threshold
    return diag_H ** (-1)

#########################################################################################################

# PT inverse
# Non- antithetic
# fixed step size
def LP_Hessian_structured_v2(F, alpha, sigma, theta_0, num_samples, time_steps, H_lambda = 1e-6, seed=1):
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
        # if prob.status == 'optimal':
        # 	H = dct_mtx @ np.diag(var_H_diag.value) @ np.transpose(dct_mtx)
        # 	# if np.linalg.det(H) == 0:
        # 	H -= H_lambda * np.identity(d)
        # else:
        # 	print("LP not optimized for the structured Hessian method.")
        # 	return None
        if not prob.status == 'optimal':
            print("LP not optimized for the structured Hessian method.")
            return None

        # **** estimate gradient (by using the method above) ****#
        y = (F_plus - F(theta_t)) / sigma
        g = LP_Gradient(y, epsilons)

        # **** update using Newton's method ****#
        theta_t -= alpha * dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value)) @ (dct_mtx @ g))

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
def LP_Hessian_structured_v3(F, alpha, sigma, theta_0, num_samples, time_steps, H_lambda=1e-6, seed=1):
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

        # **** update using Newton's method ****#
        theta_t -= alpha * dct_mtx @ (np.diag(get_PTinverse(var_H_diag.value)) @ (dct_mtx @ g))

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
# adaptive step size
def LP_Hessian_structured_v4(F, alpha, sigma, theta_0, num_samples, time_steps, H_lambda=1e-6, seed=1):
    pass





