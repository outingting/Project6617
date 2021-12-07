import numpy as np
import cvxpy as cp

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

def Gradient_LP(y, epsilons):
    """
    y = (F(theta + sigma*epsilons) - F(theta)) / sigma
    epsilons: the perturbations with UNIT VARIANCE
    """
    n, d = epsilons.shape
    
    var_z = cp.Variable(n)
    var_g = cp.Variable(d)
    obj = sum(var_z)
    constraints = [var_z >= y - epsilons @ var_g,
                   var_z >= -y + epsilons @ var_g]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.GLPK, eps=1e-6, glpk={'msg_lev': 'GLP_MSG_OFF'})
    if prob.status == 'optimal':
        return var_g.value
    return None

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

def aggregate_rollouts_hessianES(master, A, params, n_samples):
    
    all_rollouts = np.zeros([n_samples+1, 2])

    timesteps = 0
    
    # F(theta + sigma*epsilons), and F(theta - sigma*epsilons)
    assert A.shape[0] == n_samples+1
    for i in range(n_samples+1):
        w = worker(params, master, A, i)
        all_rollouts[i] = np.reshape(w.do_rollouts(), 2)
        timesteps += w.timesteps

    all_rollouts = (all_rollouts - np.mean(all_rollouts)) / (np.std(all_rollouts)  + 1e-8)
    
    # (F(theta + sigma*epsilons) - F(theta)) / sigma
    gradient_y = np.array(all_rollouts[:-1, 0] - sum(all_rollouts[-1])/2) / params["sigma"]
    # (F(theta + sigma*epsilons) + F(theta - sigma*epsilons) - 2*F(theta)) / (sigma**2)
    hessian_y = np.array(all_rollouts[:-1, 0] + all_rollouts[:-1, 1] - sum(all_rollouts[-1])) / (params["sigma"]**2)
    return(gradient_y, hessian_y, timesteps)

def aggregate_rollouts_HessianES_antithetic(master, A, params, n_samples):
    # Antithetic 
    
    all_rollouts = np.zeros([n_samples+1, 2])
    
    timesteps = 0
    
    assert A.shape[0] == n_samples+1
    for i in range(n_samples+1):
        w = worker(params, master, A, i)
        all_rollouts[i] = np.reshape(w.do_rollouts(),2)
        timesteps += w.timesteps
        
    # state normalization step?
#     all_rolouts = (all_rollouts - np.mean(all_rollouts)) / (np.std(all_rollouts) + 1e-8)
    
    raise RunTimeError('not finished')


def HessianES(params, master):
    n_samples = params['num_sensings']
    
    np.random.seed(params["seed"])
    cov = np.identity(master.N)*(params["sigma"]**2)
    mu = np.repeat(0, master.N)
    A = np.random.multivariate_normal(mu, cov, n_samples)
    A = np.vstack([A, mu]) # Adding a reference evaluation
        
    gradient_y, hessian_y, timesteps = aggregate_rollouts_hessianES(master, A, params, n_samples)
    
    g = Gradient_LP(gradient_y, A[:-1, :]/params["sigma"])
#     H = Hessian_LP(hessian_y, A[:-1, :]/params["sigma"])#-0.1*np.identity(master.N)
    H = Hessian_LP_structured(hessian_y, A[:-1, :]/params["sigma"]) - 0.1*np.identity(master.N)
#     H = -np.identity(len(g))
    try:
        Hinv = True
        update_direction = -np.linalg.inv(H)@g
    except LinAlgError:
        Hinv = False
        update_direction = g
#     if params['n_iter'] >= params['k']:
#         params['alpha'] = np.linalg.norm(np.dot(g, UUT_ort))/np.linalg.norm(np.dot(g, UUT))
    
    return(update_direction, n_samples, timesteps, Hinv)


def structured_HessianES(params, master):
    # structured hessian evolution strategy
    n_samples = params['num_sensings']
    
    np.random.seed(params['seed'])
    cov = np.identity(master.N)*(params['sigma']**2)
    mu = np.repeat(0, master.N)
    A = np.random.multivariate_normal(mu, cov, n_samples)
    A = np.vstack([A,mu])
    
    gradient_y, Hessian_y, timesteps = aggregate_rollouts_HessianES(master, A, params, n_samples)
#     gradient_y, Hessian_y, timesteps = aggregate_rollouts_HessianES_antithetic(master, A, params, n_samples)
    
    g, diag_H = Gradient_and_structured_Hessian_LP(gradient_y, Hessian_y, A[:-1, :]/params['sigma'])
    
    # dct_mtx = get_dct_mtx(master.N)
    # H = dct_mtx @ np.diag(diag_H) @ dct_mtx
    
    
