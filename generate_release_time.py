# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:26:55 2020

@author: xunzhang
"""
import math
import numpy as np

def release_time(dist,n,instance,k,r_mu_m,r_sigma_m):
    r_hat_dict = {}
    if dist == 1:
        for ins in range(instance):
            r_mu = r_mu_m[:,ins];
            r_sigma = r_sigma_m[:,ins];
            r_hat = np.zeros((n,k))
            for i in range(n):
                m = r_mu[i];
                v = r_sigma[i]*r_sigma[i];
                log_mu = math.log((m*m)/math.sqrt(v+m*m));
                log_sigma = math.sqrt(math.log(v/(m*m)+1));
                r_hat[i,:] = np.random.lognormal(log_mu,log_sigma,k);           
            r_hat_dict[ins] = r_hat
    
    if dist == 2:
        for ins in range(instance):
            r_mu = r_mu_m[:,ins];
            r_sigma = r_sigma_m[:,ins];
            r_hat = np.zeros((n,k))
            for i in range(n):
                m = r_mu[i];
                v = r_sigma[i]
                temp = np.random.normal(m,v,3*k)
                temp = list(filter(lambda x : x>=0,temp))
                r_hat[i,:] = temp[0:k]
            r_hat_dict[ins] = r_hat
#    np.sum(list(map(lambda x: x <= 0, data))) # 计算数组中小于0的个数
    
    if dist == 3:
        for ins in range(instance):
            r_mu = r_mu_m[:,ins];
            r_sigma = r_sigma_m[:,ins];
            r_hat = np.zeros((n,k))
            for i in range(n):
                m = r_mu[i]; # mean
                v = r_sigma[i]*r_sigma[i]; # variance
                A = (m*m)/v;
                B = v/m;
                r_hat[i,:] =np.random.gamma(A,B,k) 
            r_hat_dict[ins] = r_hat              
    return r_hat_dict