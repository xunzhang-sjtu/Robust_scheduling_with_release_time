# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:26:55 2020

@author: xunzhang
"""

import pathlib
import numpy as np
import pandas as pd
import computation.compute_det as det
import computation.compute_saa as saa
import computation.compute_wass as wass
import computation.compute_mom as mom
import multiprocessing as mp
import os
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import mosek_models
import pickle
import data_generation as dg


def main_process(r_mu,std_r,mu_p,std_p,n,S_train,S_test,iterations,ins,file_path,cov_bar):
    for it in range(iterations):
        print('----------------------- ins:',ins,' n:',n,' iteration:',it,'-------------------------------------')

        full_path = file_path  + 'n='+str(n)+'/' + 'ins='+str(ins)+'/'+ 'iteration='+str(it)+'/'
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
                data_info = dg.generate_correlated_Normal(mu_p,std_p,r_mu,std_r,cov_bar,S_train+S_test,0.1,0.9)
                p_bar = data_info['bar'][0:N]
                p_low = data_info['low'][0:N]
                r_bar = data_info['bar'][N:2*N]
                r_low = data_info['low'][N:2*N]
                train_data_p = (data_info['data'][0:S_train,0:N]).T
                test_data_p = (data_info['data'][S_train:S_train+S_test,0:N]).T
                train_data_r = (data_info['data'][0:S_train,N:2*N]).T
                test_data_r = (data_info['data'][S_train:S_train+S_test,N:2*N]).T
            else:
                data_info = dg.generate_Normal(mu_p,std_p,n,S_train+S_test,0.1,0.9)
                temp = data_info['data']
                p_bar = data_info['p_bar']
                p_low = data_info['p_low']
                train_data_p = temp[:,0:S_train]
                test_data_p = temp[:,S_train:S_train+S_test]

                data_info = dg.generate_Normal(r_mu,std_r,n,S_train+S_test,0.1,0.9)
                temp = data_info['data']
                r_bar = data_info['p_bar']
                r_low = data_info['p_low']
                train_data_r = temp[:,0:S_train]
                test_data_r = temp[:,S_train:S_train+S_test]
            # create a folder to store the data
            pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
            with open(full_path+'data_info.pkl', "wb") as tf:
                pickle.dump(data_info,tf)

        r_mu_esti = np.mean(train_data_r,axis = 1)
        p_mu_esti = np.mean(train_data_p,axis = 1)
        # p_std_esti = np.std(train_data,axis = 1)
        # p_mad_esti = p_std_esti/np.sqrt(np.pi/2)
        sol_det = det.deter_rand_release(n,S_test,r_mu_esti,p_mu_esti,test_data_p,test_data_r,full_path)
        sol_saa = saa.SAA_random_release(n,S_train,S_test,train_data_p,train_data_r,test_data_r,test_data_p,full_path)
        sol_wass = wass.wass_DRO_rand_release(n,train_data_r,train_data_p,test_data_r,test_data_p,p_bar,p_low,r_low,r_bar,range_c,full_path)



def effect_processing_variance(instances,iterations,n,delta_mu,delta_r,delta_ep_all,S_train,file_path):
    # # obtain a empty model
    model_DRO = mosek_models.obtain_mosek_model(S_train,n)
    models_DRO = [model_DRO.clone() for _ in range(n)] 

    for delta_ep in delta_ep_all:
        file_path1 = file_path + 'delta_ep='+str(delta_ep) + '/'
        for ins in range(instances):
            # Seed = 10 + ins
            # np.random.seed(Seed)
            mu_p = np.random.uniform(10*delta_mu,50,n)
            r_mu = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            std_p = np.sqrt(np.pi/2)*mad_p
            print('----------------------- delta_ep:',delta_ep,'-------------------------------------')
            main_process(r_mu,mu_p,std_p,n,S_train,S_test,iterations,model_DRO,models_DRO,ins,file_path1)

def effect_release_range(instances,iterations,n,delta_mu,delta_r_all,delta_ep,S_train,file_path):
    # # obtain a empty model
    model_DRO = mosek_models.obtain_mosek_model(S_train,n)
    models_DRO = [model_DRO.clone() for _ in range(n)] 

    for delta_r in delta_r_all:
        file_path1 = file_path + 'delta_r='+str(delta_r) + '/'
        for ins in range(instances):
            # Seed = 10 + ins
            # np.random.seed(Seed)
            mu_p = np.random.uniform(10*delta_mu,50,n)
            r_mu = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            std_p = np.sqrt(np.pi/2)*mad_p
            print('----------------------- delta_r:',delta_r,'-------------------------------------')
            main_process(r_mu,mu_p,std_p,n,S_train,S_test,iterations,model_DRO,models_DRO,ins,file_path1)

def effect_num_jobs(instances,iterations,delta_mu,N_all,delta_ep,S_train,file_path):

    for n in N_all:
        # obtain a empty model
        # model_DRO = mosek_models.obtain_mosek_random_model(S_train,n)
        # models_DRO = [model_DRO.clone() for _ in range(n)] 

        model_DRO = None
        models_DRO = None
        file_path1 = file_path + 'n='+str(n) + '/'
        for ins in range(instances):
            # Seed = 10 + ins
            # np.random.seed(Seed)
            mu_p = np.random.uniform(10*delta_mu,50,n)
            mu_r = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            std_p = np.sqrt(np.pi/2)*mad_p
            # ------ need to notice -----
            delta_er = delta_ep
            mad_r = np.random.uniform(0,delta_er*mu_r)
            std_r = np.sqrt(np.pi/2)*mad_r
            
            cov_bar = 0.5
            dg.generate_correlated_Normal(mu_p,std_p,mu_r,std_r,cov_bar,data_size,quan_low,quan_bar)
            
            
            print('----------------------- delta_r:',delta_r,'-------------------------------------')
            main_process(mu_r,std_r,mu_p,std_p,n,S_train,S_test,iterations,model_DRO,models_DRO,ins,file_path1)

def effect_correlation(instances,iterations,delta_mu,n,delta_ep,S_train,file_path,cov_bar_all):
    # obtain a empty model
    model_DRO = None
    models_DRO = None
    file_path1 = file_path + 'n='+str(n) + '/'
    for ins in range(instances):
        # Seed = 10 + ins
        # np.random.seed(Seed)
        mu_p = np.random.uniform(10*delta_mu,50,n)
        mu_r = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
        mad_p = np.random.uniform(0,delta_ep*mu_p)
        std_p = np.sqrt(np.pi/2)*mad_p
        # ------ need to notice -----
        delta_er = delta_ep
        mad_r = np.random.uniform(0,delta_er*mu_r)
        std_r = np.sqrt(np.pi/2)*mad_r
        
        for cov_bar in cov_bar_all:
            print('----------------------- delta_r:',delta_r,'-------------------------------------')
            main_process(mu_r,std_r,mu_p,std_p,n,S_train,S_test,iterations,ins,file_path,cov_bar)






project_path = '/Users/zhangxun/data/robust_scheduling/det_release/vns_vs_exact/'
delta_mu = 4 # control lb of mean processing time
delta_r = 0.1 # control ub of the release time
delta_ep = 1 # control the upper bound of the mad
S_train = 20
S_test = 10000
iterations = 1
instances = 20
range_c = np.arange(0,1.001,0.2)
if __name__ == '__main__':

    # # impact of variance of processing time
    # n = 20
    # file_path = '/Users/zhangxun/data/robust_scheduling/det_release/processing_variance/'
    # delta_ep_all = np.arange(0.2,2.1,0.2)
    # effect_processing_variance(instances,iterations,n,delta_mu,delta_r,delta_ep_all,S_train,file_path)


    # # impact of range of release time
    # n = 10
    # file_path = '/Users/zhangxun/data/robust_scheduling/det_release/release_range/'
    # delta_r_all = [0.05,0.1]
    # effect_release_range(instances,iterations,n,delta_mu,delta_r_all,delta_ep,S_train,file_path)


    # # impact of number of jobs
    # N_all = [3]
    # file_path = '/Users/zhangxun/data/robust_scheduling/rand_release/num_jobs/'
    # effect_num_jobs(instances,iterations,delta_mu,N_all,delta_ep,S_train,file_path)


    # impact of correlation between release and processing time

    N = 8
    cov_bar_all = np.arange(0.4,0.6,0.2)
    file_path = '/Users/zhangxun/data/robust_scheduling/rand_release/correlation/'
    effect_correlation(instances,iterations,delta_mu,N,delta_ep,S_train,file_path,cov_bar_all)

 




