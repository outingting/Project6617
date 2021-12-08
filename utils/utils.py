import numpy as np
import cvxpy as cp
from worker import worker


def get_dct_mtx(d):
    # DCT matrix
    # unitary, symmetric and real
    # Orthonormal eigenbasis for structured H
    n = 2*(d-1)
    dct_mtx = np.zeros([d,d])
    i_idx = np.array([range(n//2 )])
    i_idx = i_idx[:,1:]
    idx = 2 * np.transpose(i_idx) @ i_idx
    dct_mtx[1:-1,1:-1] = np.cos(idx*np.pi / n) * 2 / np.sqrt(d)
    for ii in range(d):
        dct_mtx[ii,0] = np.sqrt(2)
        dct_mtx[0,ii] = np.sqrt(2)
        dct_mtx[ii,-1] = (-1)**(ii) * np.sqrt(2)
        dct_mtx[-1,ii] = (-1)**(ii) * np.sqrt(2)
    dct_mtx[0, 0] = 1
    dct_mtx[0, d - 1] = 1
    dct_mtx[d - 1, 0] = 1
    dct_mtx[d - 1, d - 1] = (-1)**(d-1)
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


def Hessian_LP_structured_v2(y, epsilons):
    """
        y = (F(theta + sigma * epsilons) + F(theta - sigma * epsilons) - 2 * F(theta)) / (sigma ** 2)
        """
    # LP formulation to estimate Hessian
    # Minimizing over the space of matrices of the form
    # shown in the example 7 & 8 in the reference
    # [MATRICES DIAGONALIZED BY THE DISCRETECOSINE AND DISCRETE SINE TRANSFORMS]
    # output the 1-d array of eigenvalues of $H$

    n, d = epsilons.shape

    # Define and solve the LP for Hessian here

    var_z = cp.Variable(n)
    var_H_diag = cp.Variable(d)

    # Lower triangular mtx H
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

    #
    if prob.status == 'optimal':
        return var_H_diag.value

    return None


def aggregate_rollouts_hessianES(master, A, params, n_samples):
    all_rollouts = np.zeros([n_samples + 1, 2])

    timesteps = 0

    # F(theta + sigma*epsilons), and F(theta - sigma*epsilons)
    assert A.shape[0] == n_samples + 1
    for i in range(n_samples + 1):
        w = worker(params, master, A, i)
        all_rollouts[i] = np.reshape(w.do_rollouts(), 2)
        timesteps += w.timesteps

    all_rollouts = (all_rollouts - np.mean(all_rollouts)) / (np.std(all_rollouts) + 1e-8)

    # (F(theta + sigma*epsilons) - F(theta)) / sigma
    gradient_y = np.array(all_rollouts[:-1, 0] - sum(all_rollouts[-1]) / 2) / params["sigma"]
    # (F(theta + sigma*epsilons) + F(theta - sigma*epsilons) - 2*F(theta)) / (sigma**2)
    hessian_y = np.array(all_rollouts[:-1, 0] + all_rollouts[:-1, 1] - sum(all_rollouts[-1])) / (params["sigma"] ** 2)
    return (gradient_y, hessian_y, timesteps)


def aggregate_rollouts_HessianES_antithetic(master, A, params, n_samples):
    # Antithetic sampling case
    # n_samples for gradient estimator is 2*params['num_sensings'] antithetic samples
    # n_samples for hessian estimator is params['num_sensings'] iid samples

    all_rollouts = np.zeros([A.shape[0], 1])

    timesteps = 0

    # compute the black-box function value at \theta + \sigma * \epsilon_i and \theta for all i
    for i in range(A.shape[0]):
        w = worker(params, master, A, i)
        all_rollouts[i] = w.do_rollout()
        timesteps += w.timesteps
    # state normalization step?
    # all_rolouts = (all_rollouts - np.mean(all_rollouts)) / (np.std(all_rollouts) + 1e-8)

    # F(theta+sigma*epsilons) - F(theta) / sigma
    gradient_y = np.array(all_rollouts[:-1] - all_rollouts[-1]) / params["sigma"]
    Hessian_y = np.array(all_rollouts[:n_samples] + all_rollouts[n_samples:-1] - 2 * all_rollouts[-1]) / (
                params["sigma"] ** 2)
    return (gradient_y.reshape((-1,)), Hessian_y.reshape((-1,)), timesteps)


def get_PTinverse(diag_H, PT_threshold=-0.01):
    # Assume we are solving a maximization problem and the estimated Hessian is expected to be Negative definite
    diag_H[diag_H >= PT_threshold] = PT_threshold
    return diag_H ** (-1)


def structured_HessianES(params, master):
    # structured hessian evolution strategy
    # Apply PT inverse for matrix inversion
    # dct matrix as eigen vectors of the structured Hessian
    # Antithetic samples
    n_samples = params['num_sensings'] // 2

    np.random.seed(params['seed'])
    cov = np.identity(master.N) * (params['sigma'] ** 2)
    mu = np.repeat(0, master.N)
    A = np.random.multivariate_normal(mu, cov, n_samples)
    A = np.vstack([A, -A, mu])

    gradient_y, Hessian_y, timesteps = aggregate_rollouts_HessianES_antithetic(master, A, params, n_samples)
    g = Gradient_LP(gradient_y, A[:-1, :] / params["sigma"])
    diag_H = Hessian_LP_structured_v2(Hessian_y, A[:n_samples, :] / params["sigma"])
    dct_mtx = get_dct_mtx(master.N)
    update_direction = dct_mtx @ (np.diag(get_PTinverse(diag_H)) @ (dct_mtx @ g))

    return (update_direction, n_samples, timesteps)
