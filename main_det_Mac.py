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



def main_process(r_mu,mu_p,std_p,n,S_train,S_test,iterations,model_DRO,models_DRO,file_path):
    for it in range(iterations):
        full_path = file_path + 'iteration='+str(it)+'/'
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
            data_info = dg.generate_Normal(mu_p,std_p,n,S_train+S_test,0.1,0.9)
            temp = data_info['data']
            p_bar = data_info['p_bar']
            p_low = data_info['p_low']
            train_data = temp[:,0:S_train]
            test_data = temp[:,S_train:S_train+S_test]
            # create a folder to store the data
            pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
            # with open(full_path+'data_info.pkl', "wb") as tf:
            #     pickle.dump(data_info,tf)

        p_mu_esti = np.mean(train_data,axis = 1)
        # p_std_esti = np.std(train_data,axis = 1)
        # p_mad_esti = p_std_esti/np.sqrt(np.pi/2)
        sol_det = det.deter(n,S_test,r_mu,p_mu_esti,test_data,full_path)
        sol_saa = saa.SAA(n,S_train,S_test,train_data,r_mu,test_data,full_path)
        # sol_mom = mom.moments_DRO(n,S_test,p_mu_esti,r_mu,test_data,p_bar,p_low,full_path)
    
        exact_model = False
        sol_wass_VNS = wass.wass_DRO(n,r_mu,train_data,test_data,p_bar,p_low,sol_saa,exact_model,range_c,full_path,model_DRO,models_DRO)
        
        # exact_model = True
        # sol_wass_exact = wass.wass_DRO(n,r_mu,train_data,test_data,p_bar,p_low,sol_saa,exact_model,range_c,full_path,model_DRO,models_DRO)
        


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
        # # obtain a empty model
        # model_DRO = mosek_models.obtain_mosek_model(S_train,n)
        # models_DRO = [model_DRO.clone() for _ in range(n)] 
        model_DRO = 1
        models_DRO = 1
        file_path1 = file_path + 'n='+str(n) + '/'
        for ins in range(instances):
            # Seed = 10 + ins
            # np.random.seed(Seed)
            mu_p = np.random.uniform(10*delta_mu,50,n)
            r_mu = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            std_p = np.sqrt(np.pi/2)*mad_p
            print('----------------------- delta_r:',delta_r,'-------------------------------------')
            main_process(r_mu,mu_p,std_p,n,S_train,S_test,iterations,model_DRO,models_DRO,ins,file_path1)

def exact_vs_appro(instances,iterations,delta_mu,N_all,S_train,file_path):

    for n in N_all:
        # obtain a empty model
        # model_DRO = mosek_models.obtain_mosek_model(S_train,n)
        # models_DRO = [model_DRO.clone() for _ in range(n)] 
        model_DRO = 1
        models_DRO = 1
        
        for ins in range(instances):
            # delta_r = np.random.uniform(0.05,0.5)
            # delta_ep = np.random.uniform(0,2)
            delta_r = 0.05
            delta_ep = 1.5
            mu_p = np.random.uniform(10*delta_mu,50,n)
            r_mu = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            std_p = np.sqrt(np.pi/2)*mad_p
            file_path1 = file_path + 'n='+str(n) + '/' + 'ins='+str(ins) + '/'
            print('----------------------- n:',n,' ins:',ins,'-------------------------------------')
            main_process(r_mu,mu_p,std_p,n,S_train,S_test,iterations,model_DRO,models_DRO,file_path1)

def effect_processing_variance(instances,iterations,n,delta_mu,delta_r,delta_ep_all,S_train,file_path):
    # # obtain a empty model
    # model_DRO = mosek_models.obtain_mosek_model(S_train,n)
    # models_DRO = [model_DRO.clone() for _ in range(n)] 
    model_DRO = 1
    models_DRO = 1
    for delta_ep in delta_ep_all:
        file_path1 = file_path + 'delta_ep='+str(delta_ep) + '/'
        ins = 0
        num_cores = int(mp.cpu_count())
        p = mp.Pool(num_cores)
        rst = []
        for ins in range(instances):
            mu_p = np.random.uniform(10*delta_mu,50,n)
            mu_r = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            std_p = np.sqrt(np.pi/2)*mad_p
            print('----------------------- delta_ep:',delta_ep,' ins,',ins,'-------------------------------------')
            rst.append(p.apply_async(main_process, args=(mu_r,mu_p,std_p,n,S_train,S_test,iterations,model_DRO,models_DRO,file_path,)))
            # main_process(mu_r,mu_p,std_p,n,S_train,S_test,iterations,model_DRO,models_DRO,file_path)
        p.close()
        p.join()

        for sol in rst:
            sol.get()

# project_path = '/Users/zhangxun/data/robust_scheduling/det_release/vns_vs_exact/'
import parameters 
para = parameters.set_para()
delta_mu = para['delta_mu'] # control lb of mean processing time
delta_r = para['delta_r'] # control ub of the release time
delta_ep = para['delta_ep'] # control the upper bound of the mad
S_train = para['S_train']
S_test = para['S_test']
iterations = para['iterations']
instances = para['instances']
range_c = para['range_c']
if __name__ == '__main__':
    np.random.seed(12)
    # impact of variance of processing time
    n = 10
    file_path = 'D:/DRO_scheduling/det_release/processing_variance_RS/'
    delta_ep_all = np.arange(1.2,1.401,0.2)
    effect_processing_variance(instances,iterations,n,delta_mu,delta_r,delta_ep_all,S_train,file_path)


    # # impact of range of release time
    # n = 10
    # file_path = '/Users/zhangxun/data/robust_scheduling/det_release/release_range/'
    # delta_r_all = [0.05,0.1]
    # effect_release_range(instances,iterations,n,delta_mu,delta_r_all,delta_ep,S_train,file_path)


    # # impact of number of jobs
    # N_all = [100]
    # file_path = '/Users/zhangxun/data/robust_scheduling/det_release/num_jobs_test/'
    # effect_num_jobs(instances,iterations,delta_mu,N_all,delta_ep,S_train,file_path)


    # compare of exact and approximation
    N_all = [80]
    file_path = 'D:/DRO_scheduling/det_release/exact_vs_appro_sample_test/'
    exact_vs_appro(instances,iterations,delta_mu,N_all,S_train,file_path)

 
 




