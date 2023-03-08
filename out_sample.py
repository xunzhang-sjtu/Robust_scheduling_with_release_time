# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:11:16 2020

@author: xunzhang
"""
import numpy as np
#import generate_release_time as grt


#det_schedule
#saa_sehedule
#lldr_schedule

#iteration = 10000
#dist = 1
#n = 8
#instance = 10
#r_mu_m = 30 * n * np.random.rand(n,instance); # mean values of release time
#r_sigma_m = 0.5 * r_mu_m * np.random.rand(n,instance); # Std. deviations of release time
#
#r_hat_dict = grt.release_time(dist,n,instance,iteration,r_mu_m,r_sigma_m)
#
#p = np.ones(n)

def computeTotal(n,p_hat,r_hat,iteration,det_schedule,saa_schedule,lldr_schedule):
    det_total_list =[]
    saa_total_list =[]
    lldr_total_list =[]
    for it in range(iteration):
        r_realization = r_hat[:,it]
        p = p_hat[:,it]
        det_total = 0;
        lldr_total = 0;
        saa_total = 0;
    
        det_complete_List = np.zeros(n);
        lldr_complete_List = np.zeros(n);
        saa_complete_List = np.zeros(n);
        for i in range(n):
            det_index = int(round(det_schedule[i]))-1 # python 数组从0开始
            saa_index = int(round(saa_schedule[i]))-1
            lldr_index = int(round(lldr_schedule[i]))-1
            if i == 0:
                det_completeTime = r_realization[det_index] + p[det_index]
                det_complete_List[i] = det_completeTime
                det_total = det_total + det_completeTime
    
                saa_completeTime = r_realization[saa_index] + p[saa_index]
                saa_complete_List[i] = saa_completeTime
                saa_total = saa_total + saa_completeTime    
                
                lldr_completeTime = r_realization[lldr_index] + p[lldr_index]
                lldr_complete_List[i] = lldr_completeTime
                lldr_total = lldr_total + lldr_completeTime            
            else:
                det_completeTime = max(r_realization[det_index],det_complete_List[i-1]) + p[det_index]
                det_complete_List[i] = det_completeTime
                det_total = det_total + det_completeTime
    
                saa_completeTime = max(r_realization[saa_index],saa_complete_List[i-1]) + p[saa_index]
                saa_complete_List[i] = saa_completeTime
                saa_total = saa_total + saa_completeTime                        
    
                lldr_completeTime = max(r_realization[lldr_index],lldr_complete_List[i-1]) + p[lldr_index]
                lldr_complete_List[i] = lldr_completeTime
                lldr_total = lldr_total + lldr_completeTime            
        det_total_list.append(det_total)
        saa_total_list.append(saa_total)
        lldr_total_list.append(lldr_total)
    return det_total_list, saa_total_list, lldr_total_list

def computeTotal2(n,p_hat,r_hat,iteration,det_schedule):
    det_total_list =[]

    for it in range(iteration):
        r_realization = r_hat[:,it]
        p = p_hat[:,it]
        det_total = 0;

        det_complete_List = np.zeros(n);

        for i in range(n):
            det_index = int(round(det_schedule[i]))-1 # python 数组从0开始

            if i == 0:
                det_completeTime = r_realization[det_index] + p[det_index]
                det_complete_List[i] = det_completeTime
                det_total = det_total + det_completeTime
            
            else:
                det_completeTime = max(r_realization[det_index],det_complete_List[i-1]) + p[det_index]
                det_complete_List[i] = det_completeTime
                det_total = det_total + det_completeTime
            
        det_total_list.append(det_total)

    return det_total_list

def computeTotal_det_release(n,p_hat,r,iteration,det_schedule):
    det_total_list =[]

    for it in range(iteration):
        r_realization = r
        p = p_hat[:,it]
        det_total = 0

        det_complete_List = np.zeros(n)

        for i in range(n):
            det_index = int(round(det_schedule[i]))-1 # python 数组从0开始

            if i == 0:
                det_completeTime = r_realization[det_index] + p[det_index]
                det_complete_List[i] = det_completeTime
                det_total = det_total + det_completeTime
            
            else:
                det_completeTime = max(r_realization[det_index],det_complete_List[i-1]) + p[det_index]
                det_complete_List[i] = det_completeTime
                det_total = det_total + det_completeTime
            
        det_total_list.append(det_total)

    return det_total_list