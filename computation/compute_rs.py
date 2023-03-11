import numpy as np
from scipy.stats import truncnorm
import out_sample
import dro_models
import pickle
import pandas as pd
import mosek_models
import RS_heuristic

def RS(n,r_mu,train_data,test_data,p_bar,p_low,sol_saa):
    [N,M] = np.shape(train_data)
    # obtain a empty model
    model_mosek = mosek_models.obtain_mosek_RS_model(M,N)
    models = [model_mosek.clone() for _ in range(N)]
    def RS_given_tau(N,r_mu,M,train_data,p_bar,model_mosek,models,sol_saa,tau):
        ka_low = 0.001
        ka_bar = N+2
        while ka_bar - ka_low > 0.5:
            ka = (ka_low + ka_bar)*0.5
            sol = RS_heuristic.vns(N,r_mu,M,train_data,p_bar,model_mosek,models,sol_saa,ka)
            # print('---- tau=',tau,'ka=',ka,'obj=',sol['obj'],'--------')
            if sol['obj'] > tau:
                ka_low = ka
            else:
                ka_bar = ka

        return sol

    obj_val_saa = sol_saa['obj']
    print('-------- Solve RS --------------------')   
    tau_set = np.arange(1,2,0.05)*(obj_val_saa - r_mu.sum())
    tau_set[0] = 1.000001*(obj_val_saa - r_mu.sum())
    

    S_test = len(test_data[0,:])
    rst_RS_list = {} 
    rst_RS_time = []
    rst_RS_obj = []
    for i in range(len(tau_set)):
        tau = tau_set[i]
        sol = RS_given_tau(N,r_mu,M,train_data,p_bar,model_mosek,models,sol_saa,tau)
        # sol = dro_models.det_release_time_scheduling_RS(n,r_mu,tau_set[i],S_train,train_data,p_bar,p_low)
        rst_RS_obj.append(sol['obj'] + r_mu.sum())
        rst_RS_time.append(sol['time'])
        print('tau=',tau,'obj=',sol['obj'],'seq:',sol['x_seq'])
        rst_RS_list[tau] = np.int32(np.round(sol['x_seq'])+1) 
    tft_RS = pd.DataFrame()
    tft_RS_mean = np.zeros(len(tau_set))
    tft_RS_quan = np.zeros(len(tau_set))
    for i in range(len(tau_set)):
        tau = tau_set[i]
        tft_RS[i] = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test, rst_RS_list[tau])
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
