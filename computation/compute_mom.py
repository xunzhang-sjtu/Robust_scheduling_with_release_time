import numpy as np
from scipy.stats import truncnorm
import out_sample
import dro_models
import pickle

def moments_DRO(n,S_test,p_mu_esti,r_mu,test_data,p_bar,p_low,full_path):
    # ******** moments dro **************
    # print('-------- Solve moments DRO --------------------')
    obj_val_mom, x_seq_mom,time_mom = dro_models.det_release_time_scheduling_moments(n,p_mu_esti,r_mu,p_bar,p_low)
    x_seq_mom = np.int32(np.round(x_seq_mom)+1)
    tft_mom = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test,x_seq_mom)

    sol = {}
    sol['obj'] = obj_val_mom
    sol['seq'] = x_seq_mom
    sol['time'] = time_mom
    sol['out_obj'] = tft_mom

    print('MOM time = ',np.round(time_mom,2),\
          ',mean =',np.round(np.mean(tft_mom),2),\
            ',quantile 95 =',np.round(np.quantile(tft_mom,0.95),2))
    
    with open(full_path+'sol_mom.pkl', "wb") as tf:
        pickle.dump(sol,tf)
    return sol