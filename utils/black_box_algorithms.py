import pandas as pd
import numpy as np

import sys
sys.path.append('./asebo/')
from optimizers import Adam
from worker import worker, get_policy
from es import ES

from utils import structured_HessianES

def main_algo_asebo(params):
    # Original Asebo
    m = 0
    v = 0

    params['k'] += -1
    params['alpha'] = 1

    params['zeros'] = False
    master = get_policy(params)

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
        
        out = pd.DataFrame({'Rollouts': rollouts, 'Reward': rewards, 'Samples': samples, 'Timesteps': ts, 'Alpha': alphas})
        out.to_csv('Seed%s.csv' %(params['seed']), index=False)
        
def main_algo_structured_Hessian(params):
    # Structured hessian method
    m = 0
    v = 0

    params['k'] += -1
    params['alpha'] = 1

    params['zeros'] = False
    master = get_policy(params)

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
    
    while n_iter < params['max_iter']:
            
        params['n_iter'] = n_iter
        gradient, n_samples, timesteps = structured_HessianES(params, master)
        ts_cumulative += timesteps
        ts.append(ts_cumulative)
        alphas.append(params['alpha'])

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
        
        out = pd.DataFrame({'Rollouts': rollouts, 'Reward': rewards, 'Samples': samples, 'Timesteps': ts, 'Alpha': alphas})
        out.to_csv('Seed%s.csv' %(params['seed']), index=False)  