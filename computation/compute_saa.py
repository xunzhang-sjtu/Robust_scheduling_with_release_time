import numpy as np
from scipy.stats import truncnorm
import out_sample
import saa
import pickle
from scipy.stats import chi2


def SAA(n,S_train,S_test,train_data,r_mu,test_data,full_path):
    # ************ saa model *********************
    # print('-------- Solve SAA --------------------')
    x_seq_saa,obj_val_saa,time_saa = saa.saa_seq_det_release(n,S_train,train_data,r_mu)
    tft_saa = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test,x_seq_saa)

    tft_saa_in_sample = out_sample.computeTotal_det_release(n,train_data,r_mu,S_train,x_seq_saa)
    sol = {}
    sol['obj'] = obj_val_saa
    sol['seq'] = x_seq_saa
    sol['time'] = time_saa
    sol['out_obj'] = tft_saa
    sol['in_obj'] = tft_saa_in_sample

    print('SAA time = ',np.round(time_saa,2),\
          ',obj =',np.round(obj_val_saa,2),\
         ',mean =',np.round(np.mean(tft_saa),2),\
            ',quantile 95 =',np.round(np.quantile(tft_saa,0.95),2))
    with open(full_path+'sol_saa.pkl', "wb") as tf:
        pickle.dump(sol,tf)
    return sol

def SAA_random_release(n,S_train,S_test,train_data_p,train_data_r,test_data_r,test_data_p,full_path):
    # ************ saa model *********************
    # print('-------- Solve SAA --------------------')
    x_seq_saa,obj_val_saa,time_saa = saa.saa_seq(n,S_train,train_data_p,train_data_r)
    tft_saa = out_sample.computeTotal_rand_release(n,test_data_p,test_data_r,S_test,x_seq_saa)
    tft_saa_in_sample = out_sample.computeTotal_rand_release(n,train_data_p,train_data_r,S_train,x_seq_saa)

    sol = {}
    sol['obj'] = obj_val_saa
    sol['seq'] = x_seq_saa
    sol['time'] = time_saa
    sol['out_obj'] = tft_saa
    sol['in_obj'] = tft_saa_in_sample
    
    print('SAA time = ',np.round(time_saa,2),\
          ',obj =',np.round(obj_val_saa,2),\
         ',mean =',np.round(np.mean(tft_saa),2),\
            ',quantile 95 =',np.round(np.quantile(tft_saa,0.95),2))
    with open(full_path+'sol_saa.pkl', "wb") as tf:
        pickle.dump(sol,tf)
    return sol