# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:26:55 2020

@author: xunzhang
"""
import math
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

def main_process(r_mu,mu_p,std_p,n,S_train,S_test):
    for it in range(1):
        # temp,p_bar,p_low = generate_LogNormal(mu_p,std_p,n,S_train+S_test,0.1,0.9)
        temp,p_bar,p_low = generate_Normal(mu_p,std_p,n,S_train+S_test,0.1,0.9)

        train_data = temp[:,0:S_train]
        test_data = temp[:,S_train:S_train+S_test]
        p_mu_esti = np.mean(train_data,axis = 1)
        p_std_esti = np.std(train_data,axis = 1)
        p_mad_esti = p_std_esti/np.sqrt(np.pi/2)
        # ********** deterministic model ********************
        print('-------- Solve Det --------------------')
        x_seq_det,obj_det,time_det = det.det_seq(n,r_mu,p_mu_esti)

        # ************ saa model *********************
        print('-------- Solve SAA --------------------')
        x_seq_saa,obj_val_saa,time_saa = saa.saa_seq_det_release(n,S_train,train_data,r_mu)

        # ******** moments dro **************
        obj_val_mom, x_seq_mom,time_mom = dro_models.det_release_time_scheduling_moments(n,p_mu_esti,r_mu,p_bar,p_low)
        x_seq_mom = np.int32(np.round(x_seq_mom)+1)


        
        # ******** wassertein dro **************
        max_c = sum(p_bar - p_low)
        num_cores = int(mp.cpu_count())
        # c_set = np.arange(0,1,0.05)*max_c
        c_set = [0.5*max_c]
        

        print('-------- Solve Wass DRO --------------------')        
        p = mp.Pool(num_cores)
        rst_wass = []
        for c in c_set:
            rst_wass.append(p.apply_async(dro_models.det_release_time_scheduling_wass, args=(n,r_mu,c,S_train,train_data,p_bar,p_low,)))
            # rst_wass.append(p.apply_async(dro_models.det_release_time_scheduling_RS, args=(n,r_mu,c,S_train,train_data,p_bar,p_low,)))

        p.close()
        p.join()

        rst_wass_list = {} 
        rst_wass_time = []
        rst_wass_obj = []
        for i in range(len(c_set)):
            sol = rst_wass[i].get()
            c = sol['c']
            rst_wass_obj.append(sol['obj'] + r_mu.sum())
            rst_wass_time.append(sol['time'])
            # print('c=',c,'obj=',obj,'seq:',sol['x_seq'])
            rst_wass_list[c] = np.int32(np.round(sol['x_seq'])+1) 

        # print('-------- Solve Wass affine --------------------')        
        # p = mp.Pool(num_cores)
        # rst_affine = []
        # for c in c_set:
        #     rst_affine.append(p.apply_async(dro_det_models.det_release_time_scheduling_wass_affine, args=(n,c,S_train,train_data,p_low,p_bar,r_mu,)))
        #     # rst_wass.append(p.apply_async(dro_models.det_release_time_scheduling_RS, args=(n,r_mu,c,S_train,train_data,p_bar,p_low,)))

        # p.close()
        # p.join()

        # rst_affine_list = {} 
        # rst_affine_time = []
        # rst_affine_obj = []
        # for i in range(len(c_set)):
        #     sol = rst_affine[i].get()
        #     c = sol['c']
        #     rst_affine_obj.append(sol['obj'])
        #     rst_affine_time.append(sol['time'])
        #     # print('c=',c,'obj=',obj,'seq:',sol['x_seq'])
        #     rst_affine_list[c] = np.int32(np.round(sol['x_seq'])+1) 



        tft_det = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test-S_train,x_seq_det)
        tft_saa = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test-S_train,x_seq_saa)
        tft_mom = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test-S_train,x_seq_mom)

        tft_wass = {}
        tft_mean = np.zeros(len(c_set))
        tft_quan = np.zeros(len(c_set))

        tft_affine = {}
        tft_affine_mean = np.zeros(len(c_set))
        tft_affine_quan = np.zeros(len(c_set))


        time_df = {}
        time_df['det'] = time_det
        time_df['saa'] = time_saa
        time_df['mom'] = time_mom
        time_df['wass'] = rst_wass_time
        # time_df['affine'] = rst_affine_time

        obj_df = {}
        obj_df['det'] = obj_det
        obj_df['saa'] = obj_val_saa
        obj_df['mom'] = obj_val_mom
        obj_df['wass'] = rst_wass_obj
        # obj_df['affine'] = rst_affine_obj


        tft_df = pd.DataFrame()
        tft_df['det'] = tft_det
        tft_df['saa'] = tft_saa
        tft_df['mom'] = tft_mom

        tft_affine_df = pd.DataFrame()
        for i in range(len(c_set)):
            c = c_set[i]
            tft_wass[i] = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test-S_train, rst_wass_list[c_set[i]])
            tft_mean[i] = np.mean(tft_wass[i])
            tft_quan[i] = np.quantile(tft_wass[i],0.95)
            tft_df['wass_'+str(c)] = tft_wass[i]

            # time_df['wass_'+str(c)] = rst_wass_time[i]

            # tft_affine[i] = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test-S_train, rst_affine_list[c_set[i]])
            # tft_affine_mean[i] = np.mean(tft_affine[i])
            # tft_affine_quan[i] = np.quantile(tft_affine[i],0.95)
            # tft_affine_df['wass_'+str(c)] = tft_affine[i]

        full_path = project_path + '/data/seed='+str(Seed)+'/'+'ins='+str(ins) + '/it='+str(it) + '/S_train='+str(S_train)+'/n='+str(n)+'delta_mu='+str(delta_mu)+'delta_r='+str(delta_r)+'delta_ep='+str(delta_ep)
        pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)


        with open(full_path+'cpu_time.pkl', "wb") as tf:
            pickle.dump(time_df,tf)
        with open(full_path+'obj_df.pkl', "wb") as tf:
            pickle.dump(obj_df,tf)
        with open(full_path+'para.pkl', "wb") as tf:
            pickle.dump(para,tf)

        seq_rst = {}
        seq_rst['det'] = x_seq_det
        seq_rst['saa'] = x_seq_saa
        seq_rst['mom'] = x_seq_mom
        seq_rst['wass'] = rst_wass_list
        # seq_rst['affine'] = rst_affine_list
        with open(full_path+'seq_rst.pkl', "wb") as tf:
            pickle.dump(seq_rst,tf)
        tft_df.to_csv(full_path+'tft.csv')
        # tft_affine_df.to_csv(full_path+'tft_affine.csv')

        print('-------------------------iteration:',it)
        # print(time_df)
        print('det mean',np.round(np.mean(tft_det),2),'det quantile:',np.round(np.quantile(tft_det,0.95),2))
        print('saa mean',np.round(np.mean(tft_saa),2),'saa quantile:',np.round(np.quantile(tft_saa,0.95),2))
        print('moment mean',np.round(np.mean(tft_mom),2),'moment quantile:',np.round(np.quantile(tft_mom,0.95),2))
        print('wass mean',np.round(tft_mean,2))
        print('wass quantile:',np.round(tft_quan,2))
        # print('affine mean',np.round(tft_affine_mean,2))
        # print('affine quantile:',np.round(tft_affine_quan,2))




project_path = '/Users/zhangxun/Desktop/IJPR/sample_size_delta_r'
n = 10 # num of jobs
delta_mu = 4 # control lb of mean processing time
delta_r = 0.2 # control ub of the release time
delta_ep = 1.5 # control the upper bound of the mad
S_train = 30
S_test = 10000
iterations = 10

if __name__ == '__main__':

    for ins in range(1):

        Seed = 70 + ins
        np.random.seed(Seed)

        # mu_p = np.random.uniform(10*delta_mu,50,n)
        # r_mu = np.random.uniform(0,delta_r*mu_p.sum(),n)
        # mad_p = np.random.uniform(0,delta_ep*mu_p)
        # std_p = np.sqrt(np.pi/2)*mad_p
        # para = {}
        # para['mu_p'] = mu_p
        # para['r_mu'] = r_mu
        # para['mad_p'] = mad_p

        n_all = [10,15,20]
        for n in n_all:
            mu_p = np.random.uniform(10*delta_mu,50,n)
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            std_p = np.sqrt(np.pi/2)*mad_p

            
            delta_r_all = [0.05,0.1,0.2]
            for delta_r in delta_r_all:
                r_mu = np.random.uniform(0,delta_r*mu_p.sum(),n)
                para = {}
                para['mu_p'] = mu_p
                para['r_mu'] = r_mu
                para['mad_p'] = mad_p

                main_process(r_mu,mu_p,std_p,n,S_train,S_test)

 
 




