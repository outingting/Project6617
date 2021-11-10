import numpy as np

def ES_benchmark_gradient(F, alpha, sigma, theta_0, num_samples, time_steps):
    #****Vanilla ES method (gradient method for the Gaussian smoothing)****#
    theta_t = theta_0
    d = theta_0.shape[0]
    n = num_samples
    for t in range(time_steps):
        #**** sample epsilons ****#
        eps_list = [] 
        for i in range(n):
            eps = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d))
            eps_list.append(eps)
        #**** compute function values ****#
        F_list = []
        for i in range(n):
            F_val = F(theta_t + sigma*eps_list[i])
            F_list.append(F_val)
        #**** update theta ****#
        new_theta = theta_t
        for i in range(n):
            new_theta += alpha / (n*sigma) * F_list[i] * eps_list[i]
        theta_t = new_theta
    return theta_t, F(theta_t)

def ES_benchmark_hess_aware(F, alpha, sigma, theta_0, num_samples, time_steps, p, H_lambda):
    theta_t = theta_0
    d = theta_0.shape[0]
    n = num_samples
    H = None
    F_val_lst = []
    for t in range(time_steps):
        #**** sample epsilons ****#
        eps_list = [] 
        for i in range(n):
            eps = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d))
            eps_list.append(eps)
        #**** compute function values ****#
        F_plus_list = []
        F_minus_list = []
        F_list = []
        for i in range(n):
            F_plus_list.append(F(theta_t + sigma*eps_list[i]))
            F_minus_list.append(F(theta_t - sigma*eps_list[i]))
            F_list.append(F(theta_t))
        #**** compute Hessian every p steps ****#
        if t % p == 0:
            H = np.zeros((d,d))
            for i in range(n):
                e_i = eps_list[i].reshape((d,1))
                e_i_trans = eps_list[i].reshape((1,d))
                H += (F_plus_list[i] + F_minus_list[i] - 2*F_list[i]) * (e_i @ e_i_trans)
            H /= 2*(sigma**2)*n
            H += H_lambda * np.identity(d)
        #**** update theta: compute g ****#
        u, s, vh = np.linalg.svd(H)
        H_nh = u @ np.diag(s**-0.5) @ vh
        g = 0
        for i in range(n):
            e_i = eps_list[i].reshape((d,1)) 
            F_new = F(theta_t +  sigma* (H_nh @ e_i)  )
            g += ((F_new - F(theta_t)) / sigma ) @ (H_nh @ e_i) / n
        #**** update theta: the rest ****#
        new_theta = theta_t + alpha * g
        theta_t = new_theta
        F_val_lst.append(F(theta_t))
        
    return theta_t, F(theta_t), H, F_val_lst

def ES_hessian(F, alpha, sigma, theta_0, num_samples, time_steps, p, H_lambda):
    theta_t = theta_0
    d = theta_0.shape[0]
    n = num_samples
    H = None
    for t in range(time_steps):
        #**** sample epsilons ****#
        eps_list = [] 
        for i in range(n):
            eps = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d))
            eps_list.append(eps)
        #**** compute function values ****#
        F_list = []
        for i in range(n):
            F_val = F(theta_t + sigma*eps_list[i])
            F_list.append(F_val)
        #**** compute Hessian every p steps ****#
        if t % p == 0:
            H = np.zeros((d,d))
            for i in range(n):
                e_i = eps_list[i].reshape((d,1))
                e_i_trans = eps_list[i].reshape((1,d))
                H += F_list[i] * (np.matmul(e_i, e_i_trans) - np.identity(d) ) 
            H /= (sigma**2) * n
            H += H_lambda * np.identity(d)
        #**** update theta: compute g ****#
        u, s, vh = np.linalg.svd(H)
        H_nh = u @ np.diag(s**-0.5)
        g = 0
        for i in range(n):
            e_i = eps_list[i].reshape((d,1)) 
            F_new = F(theta_t +  sigma* (H_nh @ e_i)  )
            g += ((F_new - F(theta_t)) / sigma ) @ (H_nh @ e_i) / n
        #**** update theta: the rest ****#
        new_theta = theta_t + alpha * g
        theta_t = new_theta
        
    return theta_t, F(theta_t), H

def ES_hessian_play(F, alpha, sigma, theta_0, num_samples, time_steps, p, H_lambda):
    d = theta_0.shape[0]
    theta_t = theta_0.reshape((d,1))
    n = int(num_samples)
    H = None
    F_val_lst = []
    for t in range(time_steps):
        #**** sample epsilons ****#
        eps_list = [] 
        for i in range(n):
            eps = np.random.multivariate_normal(mean = np.zeros(d), cov = np.identity(d))
            eps_list.append(eps.reshape((d,1)))
        #**** compute function values ****#
        F_list = []
        for i in range(n):
            F_val = F(theta_t + sigma*eps_list[i])
            F_list.append(F_val)
        #**** compute Hessian every p steps ****#
        if t % p == 0:
            H = np.zeros((d,d))
            for i in range(n):
                e_i = eps_list[i]
                e_i_trans = eps_list[i].reshape((1,d))
                H += F_list[i] * (np.matmul(e_i, e_i_trans) - np.identity(d) ) 
            H /= (sigma**2) * n
#             H += H_lambda * np.identity(d)
        #**** update theta: compute g ****#
        u, s, vh = np.linalg.svd(H)
        s = s - H_lambda
        H_nh = u @ np.diag(s**-0.5) @ vh
        g = 0
        for i in range(n):
            e_i = eps_list[i].reshape((d,1))
            F_new = F(theta_t +  sigma* (H_nh @ e_i))
            g += ((F_new - F(theta_t)) / sigma ) * (H_nh @ e_i)
        #**** update theta: the rest ****#
        new_theta = theta_t + alpha * g / n
        theta_t = new_theta
        F_val_lst.append(F(theta_t))
        
        # debug 
        import pdb; pdb.set_trace()
        print(F(theta_t))
        
    return theta_t, F(theta_t), H, F_val_lst