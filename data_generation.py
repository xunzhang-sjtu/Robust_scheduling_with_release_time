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



def generate_correlated_Normal(mu_p,std_p,mu_r,std_r,cov_bar,S_train,S_test,quan_low,quan_bar):
    N = len(mu_p)
    def truncate_data(N,temp,_low,_bar):
        for i in range(N):
            tem = temp[i,:] 
            tem[tem < _low[i]] = _low[i]
            tem[tem > _bar[i]] = _bar[i]
            temp[i,:] = tem
        return temp
    
    p_bar = np.zeros(N)
    p_low = np.zeros(N)
    r_bar = np.zeros(N)
    r_low = np.zeros(N)
    train_data_p = np.zeros((N,S_train))
    test_data_p = np.zeros((N,S_test))
    train_data_r = np.zeros((N,S_train))
    test_data_r = np.zeros((N,S_test))
    for i in range(N):
        cov_mat = np.zeros((2,2))
        cov_mat[0,0] = std_p[i]*std_p[i]
        cov_mat[1,1] = std_r[i]*std_r[i]
        cov_mat[0,1] = cov_bar * std_p[i]*std_r[i]
        cov_mat[1,0] = cov_bar * std_p[i]*std_r[i]
        temp = np.random.multivariate_normal(np.asarray([mu_p[i],mu_r[i]]),cov_mat,S_train+S_test)

        p_low[i] = max(np.quantile(temp[:,0],quan_low),0.00000001)
        p_bar[i] = np.quantile(temp[:,0],quan_bar)
        train_data_p[i,:] = temp[0:S_train,0]
        test_data_p[i,:] = temp[S_train:S_train+S_test,0]


        r_low[i] = max(np.quantile(temp[:,1],quan_low),0.00000001)
        r_bar[i] = np.quantile(temp[:,1],quan_bar)
        train_data_r[i,:] = temp[0:S_train,1]
        test_data_r[i,:] = temp[S_train:S_train+S_test,1]

    train_data_p = truncate_data(N,train_data_p,p_low,p_bar)
    test_data_p = truncate_data(N,test_data_p,p_low,p_bar)
    train_data_r = truncate_data(N,train_data_r,r_low,r_bar)
    test_data_r = truncate_data(N,test_data_r,r_low,r_bar)



    return p_bar,p_low,r_bar,r_low,train_data_p,test_data_p,train_data_r,test_data_r


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
            p_bar,p_low,r_bar,r_low,train_data_p,test_data_p,train_data_r,test_data_r = generate_correlated_Normal(mu_p,std_p,r_mu,std_r,cov_bar,S_train,S_test,0.1,0.9)
            data_info = {}
            data_info['p_bar'] = p_bar
            data_info['p_low'] = p_low
            data_info['r_bar'] = r_bar
            data_info['r_low'] = r_low            
            data_info['train_data_p'] = train_data_p
            data_info['test_data_p'] = test_data_p
            data_info['train_data_r'] = train_data_r
            data_info['test_data_r'] = test_data_r

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