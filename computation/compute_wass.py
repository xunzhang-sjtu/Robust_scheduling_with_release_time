import numpy as np
from scipy.stats import truncnorm
import out_sample
import dro_models
import pickle
import heuristic
import pandas as pd
import time


def solve_dro_model(n,r_mu,c_set,S_train,train_data,p_bar,p_low,sol_saa,exact_model,model_DRO,models_DRO):
    x_saa,x_dict_saa = heuristic.decode(sol_saa['seq'])
    rst_wass_list = {} 
    rst_wass_time = []
    rst_wass_obj = []

    for i in range(len(c_set)):
        c = c_set[i]
        if exact_model:
            # ===== exact model ======
            # sol = dro_models.det_release_time_scheduling_wass(n,r_mu,1e-6,S_train,train_data,p_bar,p_low,x_saa)
            sol = dro_models.det_release_time_scheduling_RS(n,r_mu,c_set[i],S_train,train_data,p_bar)
        else: 
            # ====== heuristic solving =======
            # sol = heuristic.vns(n,r_mu,c_set[i],S_train,train_data,p_bar,model_DRO,models_DRO,sol_saa)
            # sol = dro_models.det_release_time_scheduling_wass_affine(n,c_set[i],S_train,r_mu,train_data,p_low,p_bar,x_saa)
            # sol = gold_search(n,c_set[i],S_train,r_mu,train_data,p_low,p_bar)
            sol = bisection_search(n,c_set[i],S_train,r_mu,train_data,p_low,p_bar,x_saa)
            sol['obj'] = sol['obj'] - r_mu.sum()
            print('-------------------------------------------')
            # print('Wass time = ',np.round(sol['time'],2),\
            print('Wass time = ',np.round(sol['time'],2),\
            ',obj =',np.round(sol['obj'] + r_mu.sum(),2))

        rst_wass_obj.append(sol['obj'])
        rst_wass_time.append(sol['time'])
        rst_wass_list[c] = np.int32(np.round(sol['x_seq'])+1) 

    sol_output = {}
    sol_output['obj'] = rst_wass_obj
    sol_output['seq'] = rst_wass_list
    sol_output['time'] = rst_wass_time

    return sol_output

def wass_DRO(n,r_mu,train_data,test_data,p_bar,p_low,sol_saa,exact_model,range_c,full_path,model_DRO,models_DRO):
    S_train = len(train_data[0,:])
    S_test = len(test_data[0,:])

    # # ******** wassertein dro **************
    # c_set = range_c*sol_saa['obj']
    # c_set[0] = 0.000001

    # ******** wassertein RS **************
    # sol_affine = dro_models.det_release_time_scheduling_wass_affine(n,1e-6,S_train,r_mu,train_data,p_low,p_bar,np.eye(n))
    c_set = range_c*(sol_saa['obj'] - sum(r_mu))
    # print('-------- Solve Wass DRO --------------------')        
    # solve dro model
    sol = solve_dro_model(n,r_mu,c_set,S_train,train_data,p_bar,p_low,sol_saa,exact_model,model_DRO,models_DRO)
    rst_wass_list = sol['seq']

    # --- compute out of sample performance ------
    tft_wass = pd.DataFrame()
    for i in range(len(c_set)):
        tft_wass[i] = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test, rst_wass_list[c_set[i]])
    sol['out_obj'] = tft_wass

    # print results
    if exact_model:
        print('Exact Wass time = ',np.round(sol['time'],2))
        print('Exact Wass obj = ',np.round(sol['obj'],2))
        print('Exact mean =',np.round(tft_wass.mean(axis = 0).to_list(),2))
        print('Exact quantile 95 =',np.round(tft_wass.quantile(q = 0.95,axis = 0).to_list(),2))
    else:
        print('VNS Wass time = ',np.round(sol['time'],2))
        print('VNS Wass obj = ',np.round(sol['obj'],2))
        print('VNS mean =',np.round(tft_wass.mean(axis = 0).to_list(),2))
        print('VNS quantile 95 =',np.round(tft_wass.quantile(q = 0.95,axis = 0).to_list(),2))

    # store results
    if exact_model:
        with open(full_path+'sol_wass_exact.pkl', "wb") as tf:
            pickle.dump(sol,tf)
    else:
        with open(full_path+'sol_wass_vns.pkl', "wb") as tf:
            pickle.dump(sol,tf)

    return sol


def wass_DRO_rand_release(n,train_data_r,train_data_p,test_data_r,test_data_p,p_bar,p_low,r_low,r_bar,range_c,full_path,sol_saa):
    # ******** wassertein dro **************
    max_c = sol_saa['obj']
    c_set = range_c*max_c
    c_set[0] = 0.000001
    # print('-------- Solve Wass DRO --------------------')        
    # solve dro model
    S_train = len(train_data_r[0,:])
    S_test = len(test_data_r[0,:])

    rst_wass_list = {} 
    rst_wass_time = []
    rst_wass_obj = []

    no_sol_flag = 0
    for i in range(len(c_set)):
        c = c_set[i]
        sol_wass = dro_models.rand_release_time_scheduling_wass_affine(n,c,S_train,train_data_r,train_data_p,p_low,p_bar,r_low,r_bar)
        # sol = dro_models.rand_release_time_scheduling_wass_affine_scenario(n,c,S_train,train_data_r,train_data_p,p_low,p_bar,r_low,r_bar)
        if np.isnan(sol_wass['obj']):
            no_sol_flag = 1
            break
        else:
            rst_wass_obj.append(sol_wass['obj'])
            rst_wass_time.append(sol_wass['time'])
            rst_wass_list[c] = np.int32(np.round(sol_wass['x_seq'])+1) 

        # print('Wass time = ',np.round(sol_wass['time'],2),\
        #       ' c =',c,\
        # ',obj =',np.round(sol_wass['obj'],2))

    sol = {}
    sol['no_sol_flag'] = no_sol_flag
    sol['obj'] = rst_wass_obj
    sol['seq'] = rst_wass_list
    sol['time'] = rst_wass_time

    if no_sol_flag == 0:
        # --- compute out of sample performance ------
        tft_wass = pd.DataFrame()
        for i in range(len(c_set)):
            tft_wass[i] = out_sample.computeTotal_rand_release(n,test_data_p,test_data_r,S_test,rst_wass_list[c_set[i]])
        sol['out_obj'] = tft_wass

        # print results
        print('Affine Wass time = ',np.round(sol['time'],2))
        print('Affine Wass obj = ',np.round(sol['obj'],2))
        print('Affine mean =',np.round(tft_wass.mean(axis = 0).to_list(),2))
        print('Affine quantile 95 =',np.round(tft_wass.quantile(q = 0.95,axis = 0).to_list(),2))

        # store results

        with open(full_path+'sol_wass_affine.pkl', "wb") as tf:
            pickle.dump(sol,tf)

    return sol


def bisection_search(n,tau,S_train,r_mu,train_data,p_low,p_bar,x_saa):
    total_time = 0
    a = 1e-6
    b = n+1
    f_a = {}
    f_a['obj'] = 1000000
    f_b = {}
    f_b['obj'] = 0

    sol = {}
    max_iter = 15
    iter = 0
    while b - a > 0.1 and abs(f_b['obj'] - f_a['obj']) >= 0.0001*tau and iter < max_iter:
        p = 0.5*(a+b)
        f_p = dro_models.det_release_time_scheduling_wass_affine_given_ka(n,1,S_train,r_mu,train_data,p_low,p_bar,p,x_saa)
        total_time = total_time + f_p['time']
        if f_p['obj'] <= tau:
            f_b = f_p
            b = p
            sol = f_b
            sol['c'] = b
        else:
            a = p
            f_a = f_p
            sol = f_a
            sol['c'] = b
        iter = iter + 1
        print('-----------------------------------')
        print('iter=',iter, 'ka = ',p,' obj = ',f_p['obj'],' tau = ',tau)
        sol['time'] = total_time
    return sol



def gold_search(n,c,S_train,r_mu,train_data,p_low,p_bar):

    
    a = 1e-6
    b = n+1
    
    
    t = 0.618
    p = a+(1-t)*(b-a)
    q = a+t*(b-a)
    
    
    f_a = dro_models.det_release_time_scheduling_wass_affine_given_ka(n,c,S_train,r_mu,train_data,p_low,p_bar,a)
    f_b = dro_models.det_release_time_scheduling_wass_affine_given_ka(n,c,S_train,r_mu,train_data,p_low,p_bar,b)
    f_p = dro_models.det_release_time_scheduling_wass_affine_given_ka(n,c,S_train,r_mu,train_data,p_low,p_bar,p)
    f_q = dro_models.det_release_time_scheduling_wass_affine_given_ka(n,c,S_train,r_mu,train_data,p_low,p_bar,q)
    
    print('golden search node:','a:',a,'p:',p,'q:',q,'b:',b)
    print('golden search obj:','f_a:',f_a['obj'],'f_p:',f_p['obj'],'f_q:',f_q['obj'],'f_b:',f_b['obj'])

    iterate = 5000
    loos_t = np.zeros((iterate,1))
    # step1
    k = 0
    obj_threshold = 1
    while np.abs(f_b['obj']-f_a['obj'])>=obj_threshold and np.abs(b-a) >= 0.01 and k < iterate:
        loos_t[k,0] = np.abs(f_b['obj']-f_a['obj'])
        k = k + 1
        if f_p['obj'] <= f_q['obj']:
            
            b=q
            f_b['obj']=f_q['obj']
            q=p
            f_q['obj'] =f_p['obj']
            p=a+(1-t)*(b-a)
            f_p = dro_models.det_release_time_scheduling_wass_affine_given_ka(n,c,S_train,r_mu,train_data,p_low,p_bar,p)
        else:
           
            a=p
            f_a['obj']=f_p['obj']
            p=q
            f_p['obj'] = f_q['obj']
            q=a+t*(b-a)
            f_q = dro_models.det_release_time_scheduling_wass_affine_given_ka(n,c,S_train,r_mu,train_data,p_low,p_bar,q)
        print('----------------------------------------------------------')
        print('golden search node:','a:',a,'p:',p,'q:',q,'b:',b)
        print('golden search obj:','f_a:',f_a['obj'],'f_p:',f_p['obj'],'f_q:',f_q['obj'],'f_b:',f_b['obj'])

    if f_p['obj'] <= f_q['obj']:
        f_p['ka'] = p
    else:
        f_q['ka'] = q


    return f_q