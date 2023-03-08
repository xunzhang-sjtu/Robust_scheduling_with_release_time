# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:26:55 2020

@author: xunzhang
"""


import math
import mosek


import numpy as np
from scipy.stats import truncnorm
import pandas as pd
import time
import dro_models
import out_sample
import det
import saa
import multiprocessing as mp
import os
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import pickle
import pathlib
import dro_det_models 
import heuristic
import copy



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
    
    return temp,p_bar,p_low



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


def deter(n,r_mu,p_mu_esti,test_data):
    # ********** deterministic model ********************
    print('-------- Solve Det --------------------')
    x_seq_det,obj_det,time_det = det.det_seq(n,r_mu,p_mu_esti)
    tft_det = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test-S_train,x_seq_det)
    sol = {}
    sol['obj'] = obj_det
    sol['seq'] = x_seq_det
    sol['time'] = time_det
    sol['out_obj'] = tft_det

    print('Det time = ',time_det,'mean=',np.mean(tft_det),'quantile=',np.round(np.quantile(tft_det,0.95),2))

    return sol

def SAA(n,S_train,train_data,r_mu,test_data):
    # ************ saa model *********************
    print('-------- Solve SAA --------------------')
    x_seq_saa,obj_val_saa,time_saa = saa.saa_seq_det_release(n,S_train,train_data,r_mu)
    tft_saa = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test-S_train,x_seq_saa)

    sol = {}
    sol['obj'] = obj_val_saa
    sol['seq'] = x_seq_saa
    sol['time'] = time_saa
    sol['out_obj'] = tft_saa

    print('SAA time = ',time_saa,'obj=',obj_val_saa - r_mu.sum(),'mean=',np.mean(tft_saa),'quantile=',np.round(np.quantile(tft_saa,0.95),2))

    return sol

def moments_DRO(n,p_mu_esti,r_mu,test_data,p_bar,p_low):
    # ******** moments dro **************
    print('-------- Solve moments DRO --------------------')
    obj_val_mom, x_seq_mom,time_mom = dro_models.det_release_time_scheduling_moments(n,p_mu_esti,r_mu,p_bar,p_low)
    x_seq_mom = np.int32(np.round(x_seq_mom)+1)
    tft_mom = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test-S_train,x_seq_mom)

    sol = {}
    sol['obj'] = obj_val_mom
    sol['seq'] = x_seq_mom
    sol['time'] = time_mom
    sol['out_obj'] = tft_mom

    print('MOM time = ',time_mom,'mean=',np.mean(tft_mom),'quantile=',np.round(np.quantile(tft_mom,0.95),2))

    return sol

def wass_DRO(n,r_mu,train_data,test_data,p_bar,p_low,sol_saa):
    # ******** wassertein dro **************
    max_c = sum(p_bar - p_low)
    num_cores = int(mp.cpu_count())
    c_set = np.arange(0,0.5,0.05)*max_c
    c_set[0] = 0.000001

    print('-------- Solve Wass DRO --------------------')        


    rst_wass_list = {} 
    rst_wass_time = []
    rst_wass_obj = []

    [N,M] = np.shape(train_data)
    # obtain a empty model
    model_mosek = heuristic.obtain_mosek_model(M,N)
    models = [model_mosek.clone() for _ in range(N)]


    for i in range(len(c_set)):
        # sol = {}
        # sol['obj'] = 10000000000
        # ka = n
        # while ka > 0:
        #     # sol1 = dro_models.det_release_time_scheduling_wass(n,r_mu,c_set[i],S_train,train_data,p_bar,p_low)
        #     sol2 = dro_models.det_release_time_scheduling_wass_given_ka(n,r_mu,c_set[i],S_train,train_data,p_bar,ka)

        #     if sol2['obj'] <= sol['obj']:
        #         sol = copy.deepcopy(sol2)
        #         ka = ka - 1
        #     else:
        #         break

        # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        # print('c=',c_set[i],'obj=',sol2['obj'],'seq:',sol2['x_seq'])

        # # # # sol = rst_wass[i].get()
        # sol = dro_models.det_release_time_scheduling_wass(n,r_mu,c_set[i],S_train,train_data,p_bar,p_low)
        # print('Wass obj:',sol['obj'],'Seq:',sol['x_seq'])
        sol = heuristic.vns(n,r_mu,c_set[i],S_train,train_data,p_bar,model_mosek,models,sol_saa)

        c = sol['c']
        rst_wass_obj.append(sol['obj'] + r_mu.sum())
        rst_wass_time.append(sol['time'])
        rst_wass_list[c] = np.int32(np.round(sol['x_seq'])+1) 


        # print('---------------------------------------------------')
        # print('c=',c,'exact=',sol2['obj'],'seq:',sol2['x_seq'])
        print('c=',c,'heur=',sol['obj'],'seq:',sol['x_seq'])

    tft_wass = pd.DataFrame()
    tft_mean = np.zeros(len(c_set))
    tft_quan = np.zeros(len(c_set))
    for i in range(len(c_set)):
        c = c_set[i]
        tft_wass[i] = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test-S_train, rst_wass_list[c_set[i]])
        tft_mean[i] = np.mean(tft_wass[i])
        tft_quan[i] = np.quantile(tft_wass[i],0.95)
        # tft_df['wass_'+str(c)] = tft_wass[i]


    sol = {}
    sol['obj'] = rst_wass_obj
    sol['seq'] = rst_wass_list
    sol['time'] = rst_wass_time
    sol['out_obj'] = tft_wass
    print('---------------------------------------------------')
    print('Wass time = ',rst_wass_time)
    print('mean=',np.round(tft_wass.mean(axis = 0).to_list(),2))
    print('quantile=',np.round(tft_wass.quantile(q = 0.95,axis = 0).to_list(),2))

    return sol


def RS(n,r_mu,train_data,test_data,p_bar,p_low,obj_val_saa):
    print('-------- Solve RS --------------------')   
    tau_set = np.arange(1,1.5,0.05)*(obj_val_saa - r_mu.sum())
    tau_set[0] = 1.000001*(obj_val_saa - r_mu.sum())
    
    [N,M] = np.shape(train_data)
    ka = N
    sol = dro_models.det_release_time_scheduling_RS_given_ka(n,r_mu,M,train_data,p_bar,ka)
    ka = 1
    sol = dro_models.det_release_time_scheduling_RS_given_ka(n,r_mu,M,train_data,p_bar,ka)


    rst_RS_list = {} 
    rst_RS_time = []
    rst_RS_obj = []
    for i in range(len(tau_set)):
        sol = dro_models.det_release_time_scheduling_RS(n,r_mu,tau_set[i],S_train,train_data,p_bar,p_low)
        tau = sol['c']
        rst_RS_obj.append(sol['obj'] + r_mu.sum())
        rst_RS_time.append(sol['time'])
        # print('c=',c,'obj=',sol['obj'],'seq:',sol['x_seq'])
        rst_RS_list[tau] = np.int32(np.round(sol['x_seq'])+1) 
    tft_RS = pd.DataFrame()
    tft_RS_mean = np.zeros(len(tau_set))
    tft_RS_quan = np.zeros(len(tau_set))
    for i in range(len(tau_set)):
        tau = tau_set[i]
        tft_RS[i] = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test-S_train, rst_RS_list[tau])
        tft_RS_mean[i] = np.mean(tft_RS[i])
        tft_RS_quan[i] = np.quantile(tft_RS[i],0.95)

    sol = {}
    sol['obj'] = rst_RS_obj
    sol['seq'] = rst_RS_list
    sol['time'] = rst_RS_time
    sol['out_obj'] = tft_RS

    print('RS time = ',rst_RS_time)
    print('mean=',np.round(tft_RS.mean(axis = 0).to_list(),2))
    print('quantile=',np.round(tft_RS.quantile(q = 0.95,axis = 0).to_list(),2))
    return sol


def main_process(r_mu,mu_p,std_p,n,S_train,S_test):
    for it in range(10):
        print('****************************** iteration:',it,'*************************************')
        # temp,p_bar,p_low = generate_LogNormal(mu_p,std_p,n,S_train+S_test,0.1,0.9)
        temp,p_bar,p_low = generate_Normal(mu_p,std_p,n,S_train+S_test,0.1,0.9)

        train_data = temp[:,0:S_train]
        test_data = temp[:,S_train:S_train+S_test]
        p_mu_esti = np.mean(train_data,axis = 1)
        p_std_esti = np.std(train_data,axis = 1)
        p_mad_esti = p_std_esti/np.sqrt(np.pi/2)

        sol_det = deter(n,r_mu,p_mu_esti,test_data)
        sol_saa = SAA(n,S_train,train_data,r_mu,test_data)
        # sol_mom = moments_DRO(n,p_mu_esti,r_mu,test_data,p_bar,p_low)
        # sol_RS = RS(n,r_mu,train_data,test_data,p_bar,p_low,sol_saa['obj'])
        sol_wass = wass_DRO(n,r_mu,train_data,test_data,p_bar,p_low,sol_saa)





project_path = '/Users/zhangxun/Desktop/IJPR/'
n = 10 # num of jobs
delta_mu = 4 # control lb of mean processing time
delta_r = 0.1# control ub of the release time
delta_ep = 1.5 # control the upper bound of the mad
S_train = 30
S_test = 10000
iterations = 10

if __name__ == '__main__':

    for ins in range(1):

        Seed = 10 + ins
        np.random.seed(Seed)
        n_all = [30]
        for n in n_all:
            mu_p = np.random.uniform(10*delta_mu,50,n)
            r_mu = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            std_p = np.sqrt(np.pi/2)*mad_p

            # mu_p = np.asarray([1,3,2,4])*100
            # r_mu = np.asarray([10,30,20,40])
            # mad_p = np.ones(n)*10
            # std_p = np.sqrt(np.pi/2)*mad_p

            para = {}
            para['mu_p'] = mu_p
            para['r_mu'] = r_mu
            para['mad_p'] = mad_p

            main_process(r_mu,mu_p,std_p,n,S_train,S_test)


        # n_all = [10,15,20]
        # for n in n_all:
        #     mu_p = np.random.uniform(10*delta_mu,50,n)
        #     r_mu = np.random.uniform(0,delta_r*mu_p.sum(),n)

        #     delta_ep_all = [1.5,2,2.5]
        #     for delta_ep in delta_ep_all:
        #         mad_p = np.random.uniform(0,delta_ep*mu_p)
        #         std_p = np.sqrt(np.pi/2)*mad_p
        #         para = {}
        #         para['mu_p'] = mu_p
        #         para['r_mu'] = r_mu
        #         para['mad_p'] = mad_p

        #         main_process(r_mu,mu_p,std_p,n,S_train,S_test)

 
 




