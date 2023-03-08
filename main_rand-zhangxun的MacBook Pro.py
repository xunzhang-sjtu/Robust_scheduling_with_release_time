# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:26:55 2020

@author: xunzhang
"""
import math
import numpy as np
from scipy.stats import truncnorm
import multiprocessing as mp
import time
import dro_models
import out_sample
import det
import saa
import pandas as pd
import generate_release_time as grt

import os
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import pickle
import pathlib




def generate_Normal(n,data_size,quan_low,quan_bar,mu_p,std_p):
    p_low = np.zeros(n)
    p_bar = np.zeros(n)
    temp = np.zeros((n,data_size))
    for i in range(n):
        # temp[i,:] = truncnorm.rvs((0-mu_p[i])/std_p[i],(100000000-mu_p[i])/std_p[i],mu_p[i],
        temp[i,:] = np.random.normal(mu_p[i],std_p[i],data_size)
        p_low[i] = np.quantile(temp[i,:],quan_low)
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
        r_low[i] = np.quantile(r_hat[i,:],quan_low)
        r_bar[i] = np.quantile(r_hat[i,:],quan_bar)
        
    return r_hat,r_bar,r_low

project_path = 'D:/IJPR/rand/'
n = 30 # num of jobs
delta_mu = 4 # control lb of mean processing time
delta_r = 0.1 # control ub of the release time
delta_ep = 1.5 # control the upper bound of the mad
S_train = 30
S_test = 10000
quan_low = 0.1
quan_bar = 0.9         


delta_mad_r = 0.3

if __name__ == '__main__':

    for ins in range(1):
        Seed = ins
        np.random.seed(Seed)
        for it in range(3):

            mu_p = np.random.uniform(10*delta_mu,50,n)
            mu_r = np.random.uniform(0,delta_r*mu_p.sum(),n)
            mad_p = np.random.uniform(0,delta_ep*mu_p)
            std_p = np.sqrt(np.pi/2)*mad_p
            mad_r = np.random.uniform(0,delta_ep*mu_r)
            std_r = np.sqrt(np.pi/2)*mad_r


            para = {}
            para['mu_p'] = mu_p
            para['r_mu'] = mu_r
            para['mad_p'] = mad_p
            para['mad_r'] = mad_r


            data_p,p_bar,p_low = generate_Normal(n,S_train+S_test,quan_low,quan_bar,mu_p,std_p)
            # data_p,p_bar,p_low = generate_LogNormal(mu_p,std_p,n,S_train+S_test,quan_low,quan_bar)
            data_r,r_bar,r_low = generate_Normal(n,S_train+S_test,quan_low,quan_bar,mu_r,std_r)

            train_data_p = data_p[:,0:S_train]
            test_data_p = data_p[:,S_train:S_test+S_train]

            train_data_r = data_r[:,0:S_train]
            test_data_r = data_r[:,S_train:S_test+S_train]

            p_mu_esti = np.mean(train_data_p,axis = 1)
            r_mu_esti = np.mean(train_data_r,axis = 1)
            p_std_esti = np.std(train_data_p,axis = 1)
            p_mad_esti = p_std_esti/np.sqrt(np.pi/2)


            # ********** deterministic model ********************
            print('-------- Solve Det --------------------')
            x_seq_det,obj_det,Time_det = det.det_seq(n,r_mu_esti,p_mu_esti)

            # ************ saa model *********************
            print('-------- Solve SAA --------------------')
            x_seq_saa,obj_val_saa,Time_saa = saa.saa_seq(n,S_train,train_data_p,train_data_r)


            max_c = sum(p_bar - p_low) + sum(r_bar-r_low)
            num_cores = int(mp.cpu_count())
            c_set = np.arange(0.,0.03,0.005)*max_c
            # c_set = np.asarray([0.02,0.04,0.06,0.08,0.1])*max_c
            c_set[0] = 0.00001


            

            print('-------- Solve affine DRO --------------------')
            # p = mp.Pool(num_cores)
            # rst_wass = []
            # for c in c_set:
            #     # rst_wass.append(p.apply_async(dro_models.rand_release_time_scheduling_wass, args=(n,c,S_train,train_data_r,train_data_p,p_bar,r_low,r_bar,)))
            #     rst_wass.append(p.apply_async(dro_models.rand_release_time_scheduling_wass_affine, args=(n,c,S_train,train_data_r,train_data_p,p_low,p_bar,r_low,r_bar,)))

            # p.close()
            # p.join()

            rst_wass_list = {} 
            rst_wass_time = []
            for i in range(len(c_set)):
                
                sol = dro_models.rand_release_time_scheduling_wass_affine(n,c_set[i],S_train,train_data_r,train_data_p,p_low,p_bar,r_low,r_bar)
                # sol = rst_wass[i].get()
                c = sol['c']
                obj = sol['obj']
                rst_wass_time.append(sol['time'])
                print('c=',c,'obj=',obj,'seq:',sol['x_seq'])
                rst_wass_list[c] = np.int32(np.round(sol['x_seq'])+1) 
                i = i + 1

            tft_det = out_sample.computeTotal2(n,test_data_p,test_data_r,S_test,x_seq_det)
            tft_saa = out_sample.computeTotal2(n,test_data_p,test_data_r,S_test,x_seq_saa)
            tft_wass = {}
            tft_mean = np.zeros(len(c_set))
            tft_quan = np.zeros(len(c_set))


            full_path = project_path + '/data/seed='+str(Seed)+'/'+'ins='+str(ins) + '/it='\
                +str(it) + '/S_train='+str(S_train)+'/n='+str(n)+'delta_mu='+str(delta_mu)+'delta_r='+str(delta_r)+'delta_ep='+str(delta_ep)+'delta_mad_r='+str(delta_mad_r)
            pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)

            tft_df = pd.DataFrame()
            tft_df['det'] = tft_det
            tft_df['saa'] = tft_saa
            for i in range(len(c_set)):
                c = c_set[i]
                tft_wass[i] = out_sample.computeTotal2(n,test_data_p,test_data_r,S_test,rst_wass_list[c])
                tft_mean[i] = np.mean(tft_wass[i])
                tft_quan[i] = np.quantile(tft_wass[i],0.95)
                tft_df['wass_'+str(c)] = tft_wass[i]
            # tft_df.to_csv(full_path+'tft.csv')

            cpu_time_arr = np.insert(rst_wass_time,0,Time_saa)
            cpu_time_arr = np.insert(cpu_time_arr,0,Time_det)
            # with open(full_path+'cpu_time.pkl', "wb") as tf:
            #     pickle.dump(cpu_time_arr,tf)

            # with open(full_path+'para.pkl', "wb") as tf:
            #     pickle.dump(para,tf)

            # seq_rst = {}
            # seq_rst['det'] = x_seq_det
            # seq_rst['saa'] = x_seq_saa
            # seq_rst['wass'] = rst_wass_list
            # with open(full_path+'seq_rst.pkl', "wb") as tf:
            #     pickle.dump(seq_rst,tf)



            print('****************************************')
            print('time',cpu_time_arr)

            print('det mean',np.round(np.mean(tft_det),2),'det quantile:',np.round(np.quantile(tft_det,0.95),2))
            print('saa mean',np.round(np.mean(tft_saa),2),'saa quantile:',np.round(np.quantile(tft_saa,0.95),2))
            print('----------------------------------------------')
            print('wass mean',np.round(tft_mean,2))
            print('wass quantile:',np.round(tft_quan,2))

        print(1)



