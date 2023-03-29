# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:26:55 2020

@author: xunzhang
"""
import math
import numpy as np
import pathlib
import pickle

def generate_Normal(mu_p,std_p,n,k,quan_low,quan_bar):
    p_low = np.zeros(n)
    p_bar = np.zeros(n)
    temp = np.zeros((n,k))
    for i in range(n):
        # temp[i,:] = truncnorm.rvs((0-mu_p[i])/std_p[i],(100000000-mu_p[i])/std_p[i],mu_p[i],std_p[i],k)
        temp[i,:] = np.random.normal(mu_p[i],std_p[i],k)

        p_low[i] = max(np.quantile(temp[i,:],quan_low),0.00000001)
        p_bar[i] = np.quantile(temp[i,:],quan_bar)
        

    for i in range(n):
        tem = temp[i,:] 
        tem[tem < p_low[i]] = p_low[i]
        tem[tem > p_bar[i]] = p_bar[i]
        temp[i,:] = tem
    
    data_info = {}
    data_info['data'] = temp
    data_info['p_bar'] = p_bar
    data_info['p_low'] = p_low
    return data_info



def generate_LogNormal(r_mu,r_sigma,n,k,quan_low,quan_bar):

    r_hat = np.zeros((n,k))
    r_low = np.zeros(n)
    r_bar = np.zeros(n)
    for i in range(n):
        m = r_mu[i]
        v = r_sigma[i]*r_sigma[i]
        log_mu = math.log((m*m)/math.sqrt(v+m*m))
        log_sigma = math.sqrt(math.log(v/(m*m)+1))
        r_hat[i,:] = np.random.lognormal(log_mu,log_sigma,k);           
        r_low[i] = max(np.quantile(r_hat[i,:],quan_low),0.00000001)
        r_bar[i] = np.quantile(r_hat[i,:],quan_bar)
        
    for i in range(n):
        tem = r_hat[i,:] 
        tem[tem < r_low[i]] = r_low[i]
        tem[tem > r_bar[i]] = r_bar[i]
        r_hat[i,:] = tem

    return r_hat,r_bar,r_low



def generate_correlated_Normal(mu_p,std_p,mu_r,std_r,cov_bar,data_size,quan_low,quan_bar):

    N = len(mu_p)
    mu = np.append(mu_p,mu_r)
    std = np.append(std_p*std_p,std_r*std_r)
    cov = np.diag(std)
    for i in range(N):
        temp = cov_bar * std_p[i] * std_r[i]
        cov[i,N+i] = temp
        cov[N+i,i] = temp


    temp = np.random.multivariate_normal(mu,cov,data_size)


    p_low = np.zeros(2*N)
    p_bar = np.zeros(2*N)
    for i in range(2*N):
        p_low[i] = max(np.quantile(temp[:,i],quan_low),0.00000001)
        p_bar[i] = np.quantile(temp[:,i],quan_bar)
        

    for i in range(2*N):
        tem = temp[:,i] 
        tem[tem < p_low[i]] = p_low[i]
        tem[tem > p_bar[i]] = p_bar[i]
        temp[:,i] = tem

    data_info = {}
    data_info['data'] = temp
    data_info['bar'] = p_bar
    data_info['low'] = p_low
    return data_info


def obtain_data(n,mu_p,std_p,r_mu,std_r,cov_bar,S_train,S_test,full_path):
    N = n
    # if os.path.exists(full_path+'data_info.pkl'):
    if False:
        with open(full_path+'data_info.pkl', "rb") as tf:
            data_info = pickle.load(tf)
        temp = data_info['data']
        p_bar = data_info['p_bar']
        p_low = data_info['p_low']
        train_data = temp[:,0:S_train]
        test_data = temp[:,S_train:S_train+S_test]
    else:
        # temp,p_bar,p_low = generate_LogNormal(mu_p,std_p,n,S_train+S_test,0.1,0.9)
        if not np.isnan(cov_bar):
            data_info = generate_correlated_Normal(mu_p,std_p,r_mu,std_r,cov_bar,S_train+S_test,0.1,0.9)
            p_bar = data_info['bar'][0:N]
            p_low = data_info['low'][0:N]
            r_bar = data_info['bar'][N:2*N]
            r_low = data_info['low'][N:2*N]
            train_data_p = (data_info['data'][0:S_train,0:N]).T
            test_data_p = (data_info['data'][S_train:S_train+S_test,0:N]).T
            train_data_r = (data_info['data'][0:S_train,N:2*N]).T
            test_data_r = (data_info['data'][S_train:S_train+S_test,N:2*N]).T
        else:
            data_info = generate_Normal(mu_p,std_p,n,S_train+S_test,0.1,0.9)
            temp = data_info['data']
            p_bar = data_info['p_bar']
            p_low = data_info['p_low']
            train_data_p = temp[:,0:S_train]
            test_data_p = temp[:,S_train:S_train+S_test]

            data_info = generate_Normal(r_mu,std_r,n,S_train+S_test,0.1,0.9)
            temp = data_info['data']
            r_bar = data_info['p_bar']
            r_low = data_info['p_low']
            train_data_r = temp[:,0:S_train]
            test_data_r = temp[:,S_train:S_train+S_test]
        # create a folder to store the data
        pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
        with open(full_path+'data_info.pkl', "wb") as tf:
            pickle.dump(data_info,tf)
    
    return p_bar,p_low,r_bar,r_low,train_data_p,test_data_p,train_data_r,test_data_r