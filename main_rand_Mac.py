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



def main_process(r_mu,std_r,mu_p,std_p,n,S_train,S_test,iterations,model_DRO,models_DRO,ins,file_path):
    for it in range(iterations):
        print('----------------------- ins:',ins,' n:',n,' iteration:',it,'-------------------------------------')

        full_path = file_path + 'ins='+str(ins)+'/' + 'n='+str(n)+'/' + 'iteration='+str(it)+'/'
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

        p_mu_esti = np.mean(train_data_p,axis = 1)
        # p_std_esti = np.std(train_data,axis = 1)
        # p_mad_esti = p_std_esti/np.sqrt(np.pi/2)
        # sol_det = det.deter(n,S_test,r_mu,p_mu_esti,test_data,full_path)
        # sol_saa = saa.SAA(n,S_train,S_test,train_data,r_mu,test_data,full_path)
        # sol_mom = mom.moments_DRO(n,S_test,p_mu_esti,r_mu,test_data,p_bar,p_low,full_path)
    

        import saa
        schedule,obj,saa_cpu_time = saa.saa_seq(n,S_train,train_data_p,train_data_r)

        max_c = sum(p_bar - p_low) + sum(r_bar - r_low)
        c = max_c * 0.00001
        import dro_models
        sol = dro_models.rand_release_time_scheduling_wass_affine(n,c,S_train,train_data_r,train_data_p,p_low,p_bar,r_low,r_bar)
        print(sol['obj'])
        print('-----------')
        import heuristic
        x_tem,x_dict_saa = heuristic.decode(sol['x_seq']+1)
        model_DRO.getParameter("x").setValue(x_tem)
        model_DRO.getParameter("p_bar").setValue(p_bar)
        model_DRO.getParameter("p_low").setValue(p_low)
        model_DRO.getParameter("p_hat").setValue((x_tem @ train_data_p).T)
        model_DRO.getParameter("r_bar").setValue(r_bar)
        model_DRO.getParameter("r_low").setValue(r_low)
        model_DRO.getParameter("r_hat").setValue((x_tem @ train_data_r).T)
        model_DRO.getParameter("c").setValue(c)

        model_DRO.solve()
        primal_obj = model_DRO.primalObjValue()
        # # sol = theta.level()
        # # ka = model_mosek.getVariable('ka').level()
        curr_opt_obj = primal_obj

        # exact_model = False
        # sol_wass_VNS = wass.wass_DRO(n,r_mu,train_data,test_data,p_bar,p_low,sol_saa,exact_model,range_c,full_path,model_DRO,models_DRO)
        # if n <= 20:
        #     exact_model = True
        #     sol_wass_exact = wass.wass_DRO(n,r_mu,train_data,test_data,p_bar,p_low,sol_saa,exact_model,range_c,full_path,model_DRO,models_DRO)



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
        model_DRO = mosek_models.obtain_mosek_random_model(S_train,n)
        models_DRO = [model_DRO.clone() for _ in range(n)] 


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
            print('----------------------- delta_r:',delta_r,'-------------------------------------')
            main_process(mu_r,std_r,mu_p,std_p,n,S_train,S_test,iterations,model_DRO,models_DRO,ins,file_path1)



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


    # impact of number of jobs
    N_all = [4]
    file_path = '/Users/zhangxun/data/robust_scheduling/rand_release/num_jobs/'
    effect_num_jobs(instances,iterations,delta_mu,N_all,delta_ep,S_train,file_path)



 
 




