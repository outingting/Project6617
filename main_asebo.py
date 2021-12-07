import numpy as np
import pandas as pd
import cvxpy as cp
import gym
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
import parser
import argparse


import sys
import os

sys.path.append('./utils/')
from black_box_algorithms import main_algo_asebo, main_algo_structured_Hessian

 
        

if __name__ == "__main__":
    
    # Setup for the experiment
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Swimmer-v2')
    parser.add_argument('--steps', '-s', type=int, default=1000)
    parser.add_argument('--h_dim', '-hd', type=int, default=32)
    parser.add_argument('--start', '-st', type=int, default=0)
    parser.add_argument('--max_iter', '-it', type=int, default=1000)
    parser.add_argument('--seed', '-se', type=int, default=0)

    parser.add_argument('--k', '-k', type=int, default=70)
    parser.add_argument('--num_sensings', '-sn', type=int, default=100)
    parser.add_argument('--log', '-lg', type=int, default=0)
    parser.add_argument('--threshold', '-pc', type=float, default=0.995)
    parser.add_argument('--decay', '-dc', type=float, default=0.99)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.05)
    parser.add_argument('--filename', '-f', type=str, default='')
    parser.add_argument('--policy', '-po', type=str, default='Toeplitz') # Linear or Toeplitz

    parser.add_argument('--shift', '-sh', type=int, default=0)
    parser.add_argument('--min', '-mi', type=int, default=10)
    parser.add_argument('--sigma', '-si', type=float, default=0.1)

    args = parser.parse_args()
    params = vars(args)

    params['dir'] = params['env_name'] + params['policy'] + '_h' + str(params['h_dim']) + '_lr' + str(params['learning_rate']) + '_num_sensings' + str(params['num_sensings']) +'_' + params['filename']

    if not(os.path.exists('data/'+params['dir'])):
        os.makedirs('data/'+params['dir'])
    os.chdir('data/'+params['dir'])
    
    env = gym.make(params['env_name'])
    params['ob_dim'] = env.observation_space.shape[0]
    params['ac_dim'] = env.action_space.shape[0]
    
    # Run the main algorithm
    main_algo_asebo(params)
    # main_algo_structured_Hessian(params)