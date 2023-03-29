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
        # print('----------------------- ins:',ins,' n:',n,' iteration:',it,'-------------------------------------')
        full_path = file_path + 'ins='+str(ins) + '/' 'iteration='+str(it)+'/'
        # obtain data
        p_bar,p_low,r_bar,r_low,train_data_p,test_data_p,train_data_r,test_data_r = dg.obtain_data(n,mu_p,std_p,r_mu,std_r,cov_bar,S_train,S_test,full_path)

        r_mu_esti = np.mean(train_data_r,axis = 1)
        p_mu_esti = np.mean(train_data_p,axis = 1)
        sol_det = det.deter_rand_release(n,S_test,r_mu_esti,p_mu_esti,test_data_p,test_data_r,full_path)
        sol_saa = saa.SAA_random_release(n,S_train,S_test,train_data_p,train_data_r,test_data_r,test_data_p,full_path)
        sol_wass = wass.wass_DRO_rand_release(n,train_data_r,train_data_p,test_data_r,test_data_p,p_bar,p_low,r_low,r_bar,range_c,full_path,sol_saa)
        if sol_wass['no_sol_flag'] == 1: # indicates that there may not a feasible solution
            break
    return sol_wass['no_sol_flag']


def effect_processing_variance(instances,iterations,n,delta_mu,delta_r,delta_ep_all,delta_er,S_train,file_path):
    for delta_ep in delta_ep_all:
        file_path1 = file_path + 'delta_ep='+str(delta_ep) + '/'
        ins = 0
        while ins < instances:
            mu_p = np.random.uniform(10*delta_mu,50,n)
            mu_r = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            mad_r = np.random.uniform(0,delta_er*mu_r)
            std_p = np.sqrt(np.pi/2)*mad_p
            std_r = np.sqrt(np.pi/2)*mad_r
            print('----------------------- delta_ep:',delta_ep,' ins,',ins,'-------------------------------------')
            cov_bar = np.NaN
            no_sol_flag = main_process(mu_r,std_r,mu_p,std_p,n,S_train,S_test,iterations,ins,file_path1,cov_bar)
            if no_sol_flag == 0:
                ins = ins + 1

def effect_release_variance(instances,iterations,n,delta_mu,delta_r,delta_ep,delta_er_all,S_train,file_path):
    for delta_er in delta_er_all:
        file_path1 = file_path + 'delta_er='+str(delta_er) + '/'
        ins = 0
        while ins < instances:
            mu_p = np.random.uniform(10*delta_mu,50,n)
            mu_r = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            mad_r = np.random.uniform(0,delta_er*mu_r)
            std_p = np.sqrt(np.pi/2)*mad_p
            std_r = np.sqrt(np.pi/2)*mad_r
            print('----------------------- delta_er:',delta_er,' ins,',ins,'-------------------------------------')
            cov_bar = np.NaN
            no_sol_flag = main_process(mu_r,std_r,mu_p,std_p,n,S_train,S_test,iterations,ins,file_path1,cov_bar)
            if no_sol_flag == 0:
                ins = ins + 1

def effect_both_release_and_processing_variance(instances,iterations,n,delta_mu,delta_r,delta_ep_all,delta_er_all,S_train,file_path):
    for delta_er in delta_er_all:
        for delta_ep in delta_ep_all:
            file_path1 = file_path + 'delta_er='+str(delta_er) + '/delta_ep='+str(delta_ep)+ '/'
            ins = 0
            while ins < instances:
                mu_p = np.random.uniform(10*delta_mu,50,n)
                mu_r = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
                mad_p = np.random.uniform(0,delta_ep*mu_p)
                mad_r = np.random.uniform(0,delta_er*mu_r)
                std_p = np.sqrt(np.pi/2)*mad_p
                std_r = np.sqrt(np.pi/2)*mad_r
                print('------------------- delta_er:',delta_er,' delta_ep:,',delta_ep,' ins,',ins,'-------------------------------------')
                cov_bar = np.NaN
                no_sol_flag = main_process(mu_r,std_r,mu_p,std_p,n,S_train,S_test,iterations,ins,file_path1,cov_bar)
                if no_sol_flag == 0:
                    ins = ins + 1




def effect_num_jobs(instances,iterations,delta_mu,N_all,delta_r,delta_ep,delta_er,S_train,file_path):

    for n in N_all:
        file_path1 = file_path + 'n='+str(n) + '/'
        ins = 0
        while ins < instances:
            mu_p = np.random.uniform(10*delta_mu,50,n)
            mu_p = np.random.uniform(10*delta_mu,50,n)
            mu_r = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            mad_r = np.random.uniform(0,delta_er*mu_r)
            std_p = np.sqrt(np.pi/2)*mad_p
            std_r = np.sqrt(np.pi/2)*mad_r
            print('----------------------- n:',n,' ins,',ins,'-------------------------------------')
            cov_bar = np.NaN
            no_sol_flag = main_process(mu_r,std_r,mu_p,std_p,n,S_train,S_test,iterations,ins,file_path1,cov_bar)
            if no_sol_flag == 0:
                ins = ins + 1

def effect_correlation(instances,iterations,delta_mu,n,delta_ep,delta_er,S_train,file_path,cov_bar_all):    
    for cov_bar in cov_bar_all:
        file_path1 = file_path + 'cov_bar='+str(cov_bar) + '/'
        for ins in range(instances):
            # Seed = 10 + ins
            # np.random.seed(Seed)
            mu_p = np.random.uniform(10*delta_mu,50,n)
            mu_r = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            mad_r = np.random.uniform(0,delta_er*mu_r)
            std_p = np.sqrt(np.pi/2)*mad_p
            std_r = np.sqrt(np.pi/2)*mad_r
            print('----------------------- cov_bar:',cov_bar,' ins,',ins,'-------------------------------------')
            main_process(mu_r,std_r,mu_p,std_p,n,S_train,S_test,iterations,ins,file_path1,cov_bar)







project_path = '/Users/zhangxun/data/robust_scheduling/det_release/vns_vs_exact/'
delta_mu = 4 # control lb of mean processing time
delta_r = 0.1 # control ub of the release time
delta_ep = 2 # control the upper bound of the mad
delta_er = 0.1 # control the upper bound of the mad
S_train = 20
S_test = 10000
iterations = 1
instances = 10
range_c = np.arange(0,0.3001,0.05)
if __name__ == '__main__':
    SEED = 11
    np.random.seed(SEED)
    # impact of variance of processing time
    # n = 10
    # file_path = 'D:/DRO_scheduling/rand_release/processing_variance/SEED'+str(SEED)+'/'
    # delta_ep_all = np.arange(0.2,2.1,0.2)
    # effect_processing_variance(instances,iterations,n,delta_mu,delta_r,delta_ep_all,delta_er,S_train,file_path)


    # # # impact of range of release time
    # n = 10
    # file_path = 'D:/DRO_scheduling/rand_release/release_range/SEED'+str(SEED)+'/'
    # delta_er_all = np.arange(0.0,2.1,0.2)
    # effect_release_variance(instances,iterations,n,delta_mu,delta_r,delta_ep,delta_er_all,S_train,file_path)



    # # # impact of range of release time
    # n = 10
    # file_path = 'D:/DRO_scheduling/rand_release/release_procssing_var/SEED'+str(SEED)+'/'
    # delta_er_all = np.arange(0.0,2.1,0.2)
    # delta_ep_all = np.arange(0.0,2.1,0.2)
    # effect_both_release_and_processing_variance(instances,iterations,n,delta_mu,delta_r,delta_ep_all,delta_er_all,S_train,file_path)



    # impact of number of jobs
    N_all = [10,20,30,40,50]
    # range_c = np.asarray([0.3])
    file_path = 'E:/DRO_scheduling/rand_release/num_jobs/'
    effect_num_jobs(instances,iterations,delta_mu,N_all,delta_r,delta_ep,delta_er,S_train,file_path)

    # # impact of correlation between release and processing time
    # N = 10
    # cov_bar_all = np.arange(0.2,0.8,0.2)
    # file_path = 'D:/DRO_scheduling/rand_release/correlation/'
    # effect_correlation(instances,iterations,delta_mu,N,delta_ep,delta_er,S_train,file_path,cov_bar_all)

 




