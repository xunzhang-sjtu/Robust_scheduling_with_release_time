# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:18:40 2020

@author: xunzhang
"""
import os
import pandas as pd
import numpy as np
import generate_release_time as grt
import det as det
import saa as saa
import lldr as lldr
import out_sample as out
import sche_vns as sv
#import ldr_mad as lm
import dro_exact as de
import sche_vns_det_release as svdr
import copy
import pathlib
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import time

# np.random.seed(1)
dist = 2 # 1:lognormal, 2: normal,3:gamma
n = 8 # number of jobs
instance = 10
p_cv = 0.3
r_cv = 0.3
k = 10; # number of observations
iteration = 10000


if __name__ == '__main__':
    p_cv_set = [0.3]
    for p_cv in p_cv_set:

        p_mu_m = 10 + 10 * np.random.rand(n,instance); # mean values of processing time
        # r_mu = 1 + r_cv * p_mu_m[:,0].sum() * np.random.rand(n)
        # r_mu = np.random.uniform(0,0.01*p_mu_m[:,0].sum(),n)
        r_mu_m = copy.deepcopy(p_mu_m)
        for i in range(instance):
            r_mu_m[:,i] = np.random.uniform(0,0.1*p_mu_m[:,0].sum(),n)

        r_sigma_m = r_cv * r_mu_m * np.random.rand(n,instance)

        r_hat_dict = grt.release_time(dist,n,instance,k,r_mu_m,r_sigma_m)
        # generate random release time
        r_hat_dict_out = grt.release_time(dist,n,instance,iteration,r_mu_m,r_sigma_m)


        p_sigma_m = p_cv * p_mu_m * np.random.rand(n,instance); # Std of processing time
        p_hat_dict = grt.release_time(dist,n,instance,k,p_mu_m,p_sigma_m)
        # generate random processing time
        p_hat_dict_out = grt.release_time(dist,n,instance,iteration,p_mu_m,p_sigma_m)


        det_obj_list = []
        det_seq_list = {}
        saa_obj_list = []
        saa_seq_list = {}
        lldr_obj_list = []
        dro_seq_list = {}
        dro_rand_seq_list = {}
        cpu_time = np.zeros((instance,3))


        # 计算不同方法对应的seqence
        for ins in range(instance):
            r_hat = r_hat_dict[ins]
            # r_mu_estimate = r_hat
            # r_sigma_estimate = r_hat * 0


            p_hat = p_hat_dict[ins]
            # p_mu_estimate = p_mu_m[:,0]
            # p_sigma_estimate = p_sigma_m[:,0]  
            
            # moment_2nd = p_sigma_estimate * p_sigma_estimate + p_mu_estimate * p_mu_estimate
            # det_r_obj, det_r_seq = sv.det_release_time_scheduling(n,p_mu_estimate,moment_2nd,r_mu_estimate)


            # compute det schedule and obj value
            # print('****************************solve Det model******************')
            # det_start_time = time.time()
            # det_schedule,det_obj,det_cpu_time = det.det_seq(n,r_mu_estimate,p_mu_estimate,d_mu,b,h) # deterministic approach
            # det_end_time = time.time()
            # det_cpu_time = det_end_time - det_start_time
            # det_obj_list.append(det_obj)
            # det_seq_list[ins] = det_schedule
            # x_given,x_dict = sv.decode(det_schedule,n)


            # compute saa schedule and obj value
            print('****************************solve SAA model******************')     
            saa_start_time = time.time()       
            saa_schedule,saa_obj,saa_cpu_time = saa.saa_seq(n,k,p_hat,r_hat_dict[ins])
            saa_end_time = time.time()
            saa_cpu_time = saa_end_time - saa_start_time
            saa_obj_list.append(saa_obj)
            saa_seq_list[ins] = np.asarray(saa_schedule)    



            print('******** Solve DRO model ******')
            tau = saa_obj * 1.0001
            d_bar = np.max(p_hat,axis = 1)
            d_low = np.min(p_hat,axis = 1)

            dro_start_time = time.time()
            c_set = np.arange(0,50,5)
            c_set[0] = 0.00001
            for c in c_set:
                obj_val, x_seq1 = dro_models.det_release_time_scheduling_wass(n,r_mu_m[:,ins],c,k,p_hat,d_bar,d_low)
                x_seq = np.int32(np.round(x_seq1)+1)
                print('c',c,'obj',np.round(obj_val,2),'seq',x_seq)
                # x_seq,opt_obj,time_gap = svdr.vns(n,p_mu_estimate,moment_2nd,r_mu_estimate)
                dro_end_time = time.time()
                dro_cpu_time = dro_end_time - dro_start_time
                dro_seq_list[ins,c] = x_seq

                r_bar = np.max(r_hat_dict[0],axis = 1)
                r_low = np.min(r_hat_dict[0],axis = 1) 
                obj_val_rand, x_seq1_rand = de.rand_release_time_scheduling_wass(n,c,k,r_hat_dict[ins],p_hat_dict[ins],d_bar,r_low,r_bar)
                x_seq_rand = np.int32(np.round(x_seq1_rand)+1)
                dro_rand_seq_list[ins,c] = x_seq_rand

            # cpu_time = [det_cpu_time,saa_cpu_time,dro_cpu_time]
            # cpu_time[ins]=[det_cpu_time,saa_cpu_time,dro_cpu_time]
            # print('-----------------------------------------------','p_cv',p_cv,' ins: ',ins)
            # print('det sche:', det_schedule)
            # print('saa sche:', saa_seq_list[ins])
            # print('dro sche:', dro_seq_list[ins])


        for ins in range(instance):
            
            saa_out = out.computeTotal2(n,p_hat_dict_out[ins],r_hat_dict_out[0],iteration,saa_seq_list[ins])
            dro_mean = []
            dro_quan = []
            dro_mean_rand = []
            dro_quan_rand = []
            for c in c_set:
                dro_out = out.computeTotal2(n,p_hat_dict_out[ins],r_hat_dict_out[0],iteration,dro_seq_list[ins,c])
                dro_mean.append(np.mean(dro_out))
                dro_quan.append(np.quantile(dro_out,0.95))

                dro_out_rand = out.computeTotal2(n,p_hat_dict_out[ins],r_hat_dict_out[0],iteration,dro_rand_seq_list[ins,c])
                dro_mean_rand.append(np.mean(dro_out_rand))
                dro_quan_rand.append(np.quantile(dro_out_rand,0.95))
            print('----------------------------------------------')
            print('mean ins:',ins, 'saa:',np.round(np.mean(saa_out),2),'dro:',np.round(dro_mean,2))
            print('mean ins:',ins, 'saa:',np.round(np.mean(saa_out),2),'dro rand:',np.round(dro_mean_rand,2))

            print('quantile ins:',ins, 'saa:',np.round(np.quantile(saa_out,0.95),2),'dro:',np.round(dro_quan,2))
            print('quantile ins:',ins, 'saa:',np.round(np.quantile(saa_out,0.95),2),'dro rand:',np.round(dro_quan_rand,2))

        # save file
        
        # file_pre = 'dist_' + str(dist) + '_n_' + str(n) + '_k_' + str(k) + '_r_cv_'+str(r_cv)
        # s = '/data/_p_cv_'+ str(p_cv)+'/'
        # pathlib.Path(project_path+s).mkdir(parents=True,exist_ok=True)

        # pd.DataFrame(r_mu_m).to_csv(project_path+s+file_pre+'r_mu_m.csv')
        # # pd.DataFrame(r_sigma_m).to_csv(s+'r_sigma_m.csv')
        # pd.DataFrame(quantile50).to_csv(project_path+s+file_pre+'quantile50.csv')
        # pd.DataFrame(quantile75).to_csv(project_path+s+file_pre+'quantile75.csv')
        # pd.DataFrame(quantile95).to_csv(project_path+s+file_pre+'quantile95.csv')
        # pd.DataFrame(quantile99).to_csv(project_path+s+file_pre+'quantile99.csv')
        # pd.DataFrame(cpu_time).to_csv(project_path+s+file_pre+'cpu_time.csv')
