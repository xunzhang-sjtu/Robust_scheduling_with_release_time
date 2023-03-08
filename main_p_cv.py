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
import pickle

np.random.seed(13)
n = 8 # number of jobs
instance = 10
k = 30; # number of observations
iteration = 10000


if __name__ == '__main__':

    p_mu_m = 9 + 1 * np.random.rand(n,instance); # mean values of processing time
    # r_mu = 1 + r_cv * p_mu_m[:,0].sum() * np.random.rand(n)
    r_mu = np.random.uniform(0,1,n)
    r_mu_m = copy.deepcopy(p_mu_m)
    for i in range(instance):
        r_mu_m[:,i] = r_mu
    p_cv_rand = np.random.rand(n,instance)

    p_cv_set = [0.0001,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
    for p_cv in p_cv_set:
        p_sigma_m = p_cv * p_mu_m * p_cv_rand; # Std of processing time
        # p_sigma_m = p_cv * p_mu_m # Std of processing time
        for ins in range(instance):
            p_mu_m[:,ins] = p_mu_m[:,0]
            p_sigma_m[:,ins] = p_sigma_m[:,0]

        dist = 2 # Normal distribution
        # generate training data
        p_hat_dict_Normal = grt.release_time(dist,n,instance,k,p_mu_m,p_sigma_m)
        r_hat_dict_Normal = grt.release_time(dist,n,instance,k,r_mu_m,0*r_mu_m)

        # generate training data
        p_hat_dict_LogNormal = grt.release_time(1,n,instance,k,p_mu_m,p_sigma_m)

        dist = 1 # LogNormal distribution
        # generate test data
        p_hat_dict_out = grt.release_time(dist,n,instance,iteration,p_mu_m,p_sigma_m)
        r_hat_dict_out = grt.release_time(dist,n,instance,iteration,r_mu_m,r_mu_m*0)


        s = '/effect_p_cv/p_cv='+ str(p_cv)+'/n=' + str(n) + '/'
        pathlib.Path(project_path+s).mkdir(parents=True,exist_ok=True)

        pd.DataFrame(r_mu_m).to_csv(project_path+s+'r_mu_m.csv')
        pd.DataFrame(p_mu_m).to_csv(project_path+s+'p_mu_m.csv')
        pd.DataFrame(p_sigma_m).to_csv(project_path+s+'p_sigma_m.csv')
        with open(project_path+s+'p_hat_dict_Normal.pkl', "wb") as tf:
            pickle.dump(p_hat_dict_Normal,tf)
        with open(project_path+s+'p_hat_dict_out.pkl', "wb") as tf:
            pickle.dump(p_hat_dict_out,tf)


        det_obj_list = []
        det_seq_list = {}
        saa_obj_list = []
        saa_obj_list_log = []
        saa_seq_list = {}
        saa_seq_list_log = {}
        dro_seq_list = {}
        cpu_time = np.zeros((instance,3))


        # 计算不同方法对应的seqence
        for ins in range(instance):
            r_hat = r_mu
            r_mu_estimate = r_hat
            r_sigma_estimate = r_hat * 0

            # p_hat = p_hat_dict[ins]
            p_mu_estimate = p_mu_m[:,0]
            p_sigma_estimate = p_sigma_m[:,0]
            moment_2nd = p_sigma_estimate * p_sigma_estimate + p_mu_estimate * p_mu_estimate


            # compute det schedule and obj value
            print('****************************Solve Det model******************')
            det_start_time = time.time()
            det_schedule,det_obj,det_cpu_time = det.det_seq(n,r_mu_estimate,p_mu_estimate) # deterministic approach
            det_end_time = time.time()
            det_cpu_time = det_end_time - det_start_time
            det_obj_list.append(det_obj)
            det_seq_list[ins] = det_schedule
            x_given,x_dict = sv.decode(det_schedule,n)


            # compute saa schedule and obj value
            print('****************************solve SAA model******************')     
            saa_start_time = time.time()       
            saa_schedule_Normal,saa_obj_Normal,saa_cpu_time_Normal = saa.saa_seq(n,k,p_hat_dict_Normal[ins],r_hat_dict_Normal[ins],x_given)
            saa_end_time = time.time()
            saa_cpu_time_Normal = saa_end_time - saa_start_time
            saa_obj_list.append(saa_obj_Normal)
            saa_seq_list[ins] = np.asarray(saa_schedule_Normal)    

            saa_start_time_log = time.time()       
            saa_schedule_log,saa_obj_log,saa_cpu_time_log = saa.saa_seq(n,k,p_hat_dict_LogNormal[ins],r_hat_dict_Normal[ins],x_given)
            saa_end_time_log = time.time()
            saa_cpu_time_log = saa_end_time_log - saa_start_time_log
            saa_obj_list_log.append(saa_obj_log)
            saa_seq_list_log[ins] = np.asarray(saa_schedule_log)    

    
            print('******** Solve DRO model ******')
            dro_start_time = time.time()
            if n <= 8:
                obj_val, x_seq1 = de.det_release_time_scheduling(n,p_mu_estimate,moment_2nd,r_mu_estimate,x_given)
                x_seq = np.int32(np.round(x_seq1)+1)
            else:
                x_seq,opt_obj,time_gap = svdr.vns(n,p_mu_estimate,moment_2nd,r_mu_estimate)
            dro_end_time = time.time()
            dro_cpu_time = dro_end_time - dro_start_time
            dro_seq_list[ins] = x_seq

            cpu_time = [det_cpu_time,saa_cpu_time_Normal,dro_cpu_time]
            # cpu_time[ins]=[det_cpu_time,saa_cpu_time,dro_cpu_time]
            print('-----------------------------------------------','p_cv',p_cv,' ins: ',ins)
            print('det sche:', det_schedule)
            print('saa sche:', saa_seq_list[ins])
            print('dro sche:', dro_seq_list[ins])

        with open(project_path+s+'cpu_time.pkl', "wb") as tf:
            pickle.dump(cpu_time,tf)

        with open(project_path+s+'det_seq_list.pkl', "wb") as tf:
            pickle.dump(det_seq_list,tf)
        with open(project_path+s+'saa_seq_list.pkl', "wb") as tf:
            pickle.dump(saa_seq_list,tf)
        with open(project_path+s+'saa_seq_list_log.pkl', "wb") as tf:
            pickle.dump(saa_seq_list_log,tf)
        with open(project_path+s+'dro_seq_list.pkl', "wb") as tf:
            pickle.dump(dro_seq_list,tf)

        
        for ins in range(instance):
            tft_out = pd.DataFrame()
            det_tft_list = out.computeTotal2(n,p_hat_dict_out[ins],r_hat_dict_out[0],iteration,det_seq_list[ins])
            saa_tft_list_log = out.computeTotal2(n,p_hat_dict_out[ins],r_hat_dict_out[0],iteration,saa_seq_list_log[ins])
            saa_tft_list_normal = out.computeTotal2(n,p_hat_dict_out[ins],r_hat_dict_out[0],iteration,saa_seq_list[ins])
            dro_tft_list = out.computeTotal2(n,p_hat_dict_out[ins],r_hat_dict_out[0],iteration,dro_seq_list[ins])


            print('----------------------------------------------------------')
            print('ins',ins,'saa log avg',np.mean(saa_tft_list_log)/np.mean(det_tft_list),\
                    'saa normal avg',np.mean(saa_tft_list_normal)/np.mean(det_tft_list),\
                    'dro avg',np.mean(dro_tft_list)/np.mean(det_tft_list))
            print('ins',ins,'saa log 95',np.quantile(saa_tft_list_log,0.95)/np.quantile(det_tft_list,0.95),\
                    'saa normal 95',np.quantile(saa_tft_list_normal,0.95)/np.quantile(det_tft_list,0.95),\
                    'dro 95',np.quantile(dro_tft_list,0.95)/np.quantile(det_tft_list,0.95))

            tft_out['det'] = det_tft_list
            tft_out['saa_log'] = saa_tft_list_log
            tft_out['saa_normal'] = saa_tft_list_normal
            tft_out['dro'] = dro_tft_list

            tft_out.to_pickle(project_path+s+'tft_out_ins='+str(ins)+'.pkl')

        # quantile99 = np.zeros((instance,3))
        # quantile95 = np.zeros((instance,3))
        # quantile75 = np.zeros((instance,3))
        # quantile50 = np.zeros((instance,3))
        # for ins in range(instance):
        #     det_total_list, saa_total_list, lldr_total_list = out.computeTotal(n,p_hat_dict_out[ins],r_hat_dict_out[0],iteration,det_seq_list[ins],saa_seq_list[ins],dro_seq_list[ins])
            
        #     det99 = np.percentile(det_total_list,99)
        #     det95 = np.percentile(det_total_list,95)
        #     det75 = np.percentile(det_total_list,75)
        #     det50 = np.mean(det_total_list)
            
        #     saa99 = np.percentile(saa_total_list,99)
        #     saa95 = np.percentile(saa_total_list,95)
        #     saa75 = np.percentile(saa_total_list,75)
        #     saa50 = np.mean(saa_total_list)    

        #     lldr99 = np.percentile(lldr_total_list,99)
        #     lldr95 = np.percentile(lldr_total_list,95)
        #     lldr75 = np.percentile(lldr_total_list,75)
        #     lldr50 = np.mean(lldr_total_list)
            
        #     quantile99[ins,:] = [det99,saa99,lldr99]
        #     quantile95[ins,:] = [det95,saa95,lldr95]
        #     quantile75[ins,:] = [det75,saa75,lldr75]
        #     quantile50[ins,:] = [det50,saa50,lldr50]

        # avg_quantile_99 = quantile99.mean(axis = 0)
        # avg_quantile_95 = quantile95.mean(axis = 0)
        # avg_quantile_75 = quantile75.mean(axis = 0)
        # avg_quantile_50 = quantile50.mean(axis = 0)
        # avg_cpu_time = np.mean(cpu_time,axis = 0)

        # print('mean saa_gap:',(avg_quantile_50[0] - avg_quantile_50[1])/avg_quantile_50[0])
        # print('mean dro_gap:',(avg_quantile_50[0] - avg_quantile_50[2])/avg_quantile_50[0])
        # print('95 saa_gap:',(avg_quantile_95[0] - avg_quantile_95[1])/avg_quantile_95[0])
        # print('95 dro_gap:',(avg_quantile_95[0] - avg_quantile_95[2])/avg_quantile_95[0])

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