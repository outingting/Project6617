import numpy as np
import pandas as pd
import cvxpy as cp
import gym
import os
from sklearn.linear_model import LinearRegression
from numpy.linalg import LinAlgError
from scipy.linalg import cholesky
from scipy.stats import ortho_group
from numpy.random import standard_normal
from asebo.worker import get_policy, worker
from asebo.es import ES
from asebo.optimizers import Adam



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
    epsilons: the perturbations with UNIT VARIANCE
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
def get_PTinverse(diag_H, PT_threshold=-1):
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

def Hessian_LP_structured(y, epsilons, new=True):
    """
    y = (F(theta + sigma * epsilons) + F(theta - sigma * epsilons) - 2 * F(theta)) / (sigma ** 2)
    """
    # LP formulation to estimate Hessian
    # Minimizing over the space of matrices of the form
    # shown in the example 7 & 8 in the reference
    # [MATRICES DIAGONALIZED BY THE DISCRETECOSINE AND DISCRETE SINE TRANSFORMS]
    
    n, d = epsilons.shape
    
    if new:
        var_H_diag = cp.Variable(d)
        dct_mtx = get_dct_mtx(d)

        Uv_sq = cp.square(epsilons@dct_mtx)
        yhat = Uv_sq @ var_H_diag
        constraints = [var_H_diag <= 0]
        prob = cp.Problem(cp.Minimize(cp.norm(yhat-y, 1)), constraints)
        prob.solve(solver=cp.GUROBI)
    else:
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
        # H = dct_mtx @ np.diag(var_H_diag.value) @ np.transpose(dct_mtx)
        return var_H_diag.value, dct_mtx
    else: 
        print("LP not optimized for the structured Hessian method. Problem Status:", prob.status) 
        return None

def aggregate_rollouts_hessianES(master, A, params):
    """
    Inputs:
        A: a matrix of perturbations to try
    Output:
        all_rollouts: n x 2 matrix, where n is the number of rows in A
    """
    # F(theta + sigma*epsilons), and F(theta - sigma*epsilons)
    n = A.shape[0]
    all_rollouts = np.zeros([n, 2])
    timesteps = 0
    for i in range(n):
        w = worker(params, master, A, i)
        all_rollouts[i] = np.reshape(w.do_rollouts(), 2)
        timesteps += w.timesteps

    all_rollouts = (all_rollouts - np.mean(all_rollouts)) / (np.std(all_rollouts)  + 1e-8)
    return all_rollouts, timesteps

def gradient_LP_estimator(all_rollouts, A, sigma, SigmaInv=None):
    # (F(theta + sigma*epsilons) - F(theta)) / sigma
    gradient_y = np.array(all_rollouts[:-1, 0] - sum(all_rollouts[-1])/2) / sigma
    g = Gradient_LP(gradient_y, A[:-1, :]/sigma)
    
    return g
def gradient_LP_antithetic_estimator(all_rollouts, A, sigma, SigmaInv=None):
    gradient_y = np.array(
            np.concatenate([all_rollouts[:-1, 0], all_rollouts[:-1, 1]])
            - sum(all_rollouts[-1])/2
        ) / sigma
    epsilons = np.vstack([A[:-1, :]/sigma,-A[:-1, :]/sigma])
    g = Gradient_LP(gradient_y, epsilons)
    return g
def gradient_L2_antithetic_estimator(all_rollouts, A, sigma, SigmaInv=None):
    gradient_y = np.array(
            np.concatenate([all_rollouts[:-1, 0], all_rollouts[:-1, 1]])
            - sum(all_rollouts[-1])/2
        ) / sigma
    epsilons = np.vstack([A[:-1, :]/sigma,-A[:-1, :]/sigma])
    g = Gradient_L2(gradient_y, epsilons)
    return g

def gradient_antithetic_estimator(all_rollouts, A, sigma, SigmaInv=None):
    _, d = A.shape
    if SigmaInv is None:
        SigmaInv = np.identity(d)
    gradient_y = np.array(
            np.concatenate([all_rollouts[:-1, 0], all_rollouts[:-1, 1]])
            )
    epsilons = np.vstack([A[:-1, :]/sigma,-A[:-1, :]/sigma])
    g = (gradient_y@(epsilons@SigmaInv)) / sigma / len(gradient_y)
    return g
    
def invHessian_LP_estimator(all_rollouts, A, sigma, H_lambda=1):
    # (F(theta + sigma*epsilons) + F(theta - sigma*epsilons) - 2*F(theta)) / (sigma**2)
    _, d = A.shape
    hessian_y = np.array(all_rollouts[:-1, 0] + all_rollouts[:-1, 1] - sum(all_rollouts[-1])) / (sigma**2)
    H = Hessian_LP(hessian_y, A[:-1, :]/sigma) - H_lambda*np.identity(d)
    
    try:
        Hinv = np.linalg.inv(H)
    except LinAlgError:
        Hinv = -np.identity(d)
        
    return Hinv

def invHessian_LP_structured_PTinv_estimator(all_rollouts, A, sigma, PT_threshold=-1):
    # (F(theta + sigma*epsilons) + F(theta - sigma*epsilons) - 2*F(theta)) / (sigma**2)
    _, d = A.shape
    hessian_y = np.array(all_rollouts[:-1, 0] + all_rollouts[:-1, 1] - sum(all_rollouts[-1])) / (sigma**2)
    var_H_diag, dct_mtx = Hessian_LP_structured(hessian_y, A[:-1, :]/sigma)
    Hinv = dct_mtx @ (np.diag(get_PTinverse(var_H_diag, PT_threshold)) @ dct_mtx)
    return Hinv


def invHessian_L2_structured_PTinv_estimator(all_rollouts, A, sigma, PT_threshold=-1):
    # (F(theta + sigma*epsilons) + F(theta - sigma*epsilons) - 2*F(theta)) / (sigma**2)
    _, d = A.shape
    hessian_y = np.array(all_rollouts[:-1, 0] + all_rollouts[:-1, 1] - sum(all_rollouts[-1])) / (sigma**2)
    var_H_diag, dct_mtx = Hessian_L2_structured(hessian_y, A[:-1, :]/sigma)
    Hinv = dct_mtx @ (np.diag(get_PTinverse(var_H_diag, PT_threshold)) @ dct_mtx)
    
    return Hinv

def invHessian_identity_estimator(all_rollouts, A, sigma, H_lambda=1):
    # (F(theta + sigma*epsilons) + F(theta - sigma*epsilons) - 2*F(theta)) / (sigma**2)
    _,d = A.shape
    Hinv = -np.identity(d)
    return Hinv

def HessianES(params, master, gradient_estimator, invhessian_estimator):
    n_samples = params['num_sensings']    
    cov = np.identity(master.N)
    mu = np.repeat(0, master.N)
    A = np.random.multivariate_normal(mu, cov, n_samples)
    A *= params["sigma"]
    A = np.vstack([A, mu]) # Adding a reference evaluation
        
    rollouts, timesteps = aggregate_rollouts_hessianES(master, A, params)
    
    g = gradient_estimator(rollouts, A, params["sigma"])
    invH = invhessian_estimator(rollouts, A, params["sigma"])

    return(g, invH, n_samples, timesteps)

def orthogonal_gaussian(d, n_samples):
    blocks = n_samples//d + (n_samples%d >0)
    out = np.concatenate([ortho_group.rvs(d) for _ in range(blocks)])[:n_samples]
    norms = np.sqrt(np.random.chisquare(d, size=n_samples))
    out = (out.T * norms).T
    return out

    

def HessianESv2(params, master, gradient_estimator, invhessian_estimator, cov=None):
    """
    Samples from invHessian of previous round
    """
    n_samples = params['num_sensings']    
    if cov is None:
        cov = np.identity(master.N)
    mu = np.repeat(0, master.N)

    # A = np.random.multivariate_normal(mu, cov, n_samples)
    A = orthogonal_gaussian(master.N, n_samples)
    # A /= np.linalg.norm(A, axis =-1)[:, np.newaxis]
    A *= params["sigma"] 
    A = np.vstack([A, mu]) # Adding a reference evaluation
        
    rollouts, timesteps = aggregate_rollouts_hessianES(master, A, params)
    
    g = gradient_estimator(rollouts, A, params["sigma"], np.linalg.inv(cov))
    invH = invhessian_estimator(rollouts, A, params["sigma"])
    return(g, invH, n_samples, timesteps)


def run_HessianES(params, gradient_estimator, invhessian_estimator, master=None):
    params['dir'] = params['env_name'] + params['policy'] + '_h' + str(params['h_dim']) + '_lr' + str(params['learning_rate']) \
                    + '_num_sensings' + str(params['num_sensings']) +'_' + 'sampleFromInvH'+ str(params['sample_from_invH'])
    data_folder = './data/'+params['dir']+'_hessianES'
    if not(os.path.exists(data_folder)):
        os.makedirs(data_folder)
    if not master:
        master = get_policy(params)
    print("Policy Dimension: ", master.N)

    test_policy = worker(params, master, np.zeros([1, master.N]), 0)
    reward = test_policy.rollout(train=False)

    n_eps = 0
    n_iter = 1
    ts_cumulative = 0
    ts = [0]
    rollouts = [0]
    rewards = [reward]
    samples = [0]
    cov = np.identity(master.N)
    np.random.seed(params['seed'])
    while n_iter < params['max_iter']:
        params['n_iter'] = n_iter
        g, invH, n_samples, timesteps = HessianESv2(params, master, gradient_estimator, invhessian_estimator, cov)
        if params['sample_from_invH']:
            cov = -invH
        update_direction = -invH@g
        lr = params['learning_rate']
        # Backtracking
        if params['backtracking']:
            update = lr*update_direction
            master.update(update)
            # Evaluate
            test_policy = worker(params, master, np.zeros([1, master.N]), 0)
            reward = test_policy.rollout(train=False)
            count = 0
            while (reward < rewards[-1] + lr*params['alpha']*(g@update_direction)) and lr > 1e-20:
                count += 1
                master.update(-update) # Cancel the previous update first
                lr *= params['beta']
                update = lr*update_direction
                master.update(update)
                # Evaluate
                test_policy = worker(params, master, np.zeros([1, master.N]), 0)
                reward = test_policy.rollout(train=False)
                timesteps += test_policy.timesteps
            # if (reward < rewards[-1] + lr*params['alpha']*(g@update_direction)):
            #     # Do not update
            #     master.update(-update)
            #     reward = rewards[-1]

        else: 
            update = lr*update_direction
            master.update(update)
            # Evaluate
            test_policy = worker(params, master, np.zeros([1, master.N]), 0)
            reward = test_policy.rollout(train=False)



        # Book keeping
        ts_cumulative += timesteps 
        ts.append(ts_cumulative)

        n_eps += 2 * n_samples
        rollouts.append(n_eps)

        rewards.append(reward)
        samples.append(n_samples)

        print('Iteration: %s, Leanring Rate: %.3e, Rollouts: %s, Reward: %.2f, Update Direction Norm: %.2f' %(n_iter, lr, n_eps, reward,  np.linalg.norm(update_direction)))
        n_iter += 1

        out = pd.DataFrame({'Rollouts': rollouts, 'Learning Rate':
        lr, 'Reward': rewards, 'Samples': samples,
        'Timesteps': ts})
        out.to_csv('%s/Seed%s.csv' %(data_folder, params['seed']),
        index=False)   

        np.save("{}/hessianES_params_seed{}.npy".format(data_folder, params['seed']),
        master.params)

    return ts, rewards, master




def run_asebo(params, master=None):
    params['dir'] = params['env_name'] + params['policy'] + '_h' + str(params['h_dim']) + '_lr' + str(params['learning_rate']) \
                    + '_num_sensings' + str(params['num_sensings']) +'_' + 'sampleFromInvH'+ str(params['sample_from_invH'])
    data_folder = './data/'+params['dir']+'_asebo'
    if not(os.path.exists(data_folder)):
        os.makedirs(data_folder) 

    env = gym.make(params['env_name'])
    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]
    
    m = 0
    v = 0

    params['k'] += -1
    params['alpha'] = 1
        
    params['zeros'] = False
    if not master:
        master = get_policy(params)
    print("Policy Dimension: ", master.N)
    
    if params['log']:
        params['num_sensings'] = 4 + int(3 * np.log(master.N))
    
    if params['k'] > master.N:
        params['k'] = master.N
        
    n_eps = 0
    n_iter = 1
    ts_cumulative = 0
    ts = []
    rollouts = []
    rewards = []
    samples = []
    alphas = []
    G = []
        
    while n_iter < params['max_iter']:
            
        params['n_iter'] = n_iter
        gradient, n_samples, timesteps = ES(params, master, G)
        ts_cumulative += timesteps
        ts.append(ts_cumulative)
        alphas.append(params['alpha'])

        if n_iter == 1:
            G = np.array(gradient)
        else:
            G *= params['decay']
            G = np.vstack([G, gradient])
        n_eps += 2 * n_samples
        rollouts.append(n_eps)
        gradient /= (np.linalg.norm(gradient) / master.N + 1e-8)
            
        update, m, v = Adam(gradient, m, v, params['learning_rate'], n_iter)
            
        master.update(update)
        test_policy = worker(params, master, np.zeros([1, master.N]), 0)
        reward = test_policy.rollout(train=False)
        rewards.append(reward)
        samples.append(n_samples)
            
        print('Iteration: %s, Rollouts: %s, Reward: %s, Alpha: %s, Samples: %s' %(n_iter, n_eps, reward, params['alpha'], n_samples))
        n_iter += 1


        out = pd.DataFrame({'Rollouts': rollouts, 'Learning Rate':
        params['learning_rate'], 'Reward': rewards, 'Samples': samples,
        'Timesteps': ts, 'Alpha': alphas})
        out.to_csv('%s/Seed%s.csv' %(data_folder, params['seed']),
        index=False)   

        np.save("{}/asebo_params_seed{}.npy".format(data_folder, params['seed']),
        master.params)
    return ts, rewards, master