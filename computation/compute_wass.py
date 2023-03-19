import numpy as np
from scipy.stats import truncnorm
import out_sample
import dro_models
import pickle
import heuristic
import pandas as pd

def solve_dro_model(n,r_mu,c_set,S_train,train_data,p_bar,p_low,sol_saa,exact_model,model_DRO,models_DRO):
    x_saa,x_dict_saa = heuristic.decode(sol_saa['seq'])
    rst_wass_list = {} 
    rst_wass_time = []
    rst_wass_obj = []

    for i in range(len(c_set)):
        if exact_model:
            # ===== exact model ======
            sol = dro_models.det_release_time_scheduling_wass(n,r_mu,c_set[i],S_train,train_data,p_bar,p_low,x_saa)
        else: 
            # ====== heuristic solving =======
            sol = heuristic.vns(n,r_mu,c_set[i],S_train,train_data,p_bar,model_DRO,models_DRO,sol_saa)
        c = sol['c']
        rst_wass_obj.append(sol['obj'] + r_mu.sum())
        rst_wass_time.append(sol['time'])
        rst_wass_list[c] = np.int32(np.round(sol['x_seq'])+1) 






        # print('Wass time = ',np.round(sol['time'],2),\
        # ',obj =',np.round(sol['obj'] + r_mu.sum(),2))
    sol = {}
    sol['obj'] = rst_wass_obj
    sol['seq'] = rst_wass_list
    sol['time'] = rst_wass_time

    return sol

def wass_DRO(n,r_mu,train_data,test_data,p_bar,p_low,sol_saa,exact_model,range_c,full_path,model_DRO,models_DRO):
    # ******** wassertein dro **************
    max_c = sum(p_bar - p_low)
    c_set = range_c*max_c
    c_set[0] = 0.000001
    # print('-------- Solve Wass DRO --------------------')        
    # solve dro model
    S_train = len(train_data[0,:])
    S_test = len(test_data[0,:])
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

    for i in range(len(c_set)):
        c = c_set[i]
        sol = dro_models.rand_release_time_scheduling_wass_affine(n,c,S_train,train_data_r,train_data_p,p_low,p_bar,r_low,r_bar)
        # sol = dro_models.rand_release_time_scheduling_wass_affine_scenario(n,c,S_train,train_data_r,train_data_p,p_low,p_bar,r_low,r_bar)

        
        c = sol['c']
        rst_wass_obj.append(sol['obj'])
        rst_wass_time.append(sol['time'])
        rst_wass_list[c] = np.int32(np.round(sol['x_seq'])+1) 

        # print('Wass time = ',np.round(sol['time'],2),\
        # ',obj =',np.round(sol['obj'] + r_mu.sum(),2))
    sol = {}
    sol['obj'] = rst_wass_obj
    sol['seq'] = rst_wass_list
    sol['time'] = rst_wass_time



    

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