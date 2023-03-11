import numpy as np
from scipy.stats import truncnorm
import out_sample
import det
import pickle


def deter(n,S_test,r_mu,p_mu_esti,test_data,full_path):
    # ********** deterministic model ********************
    # print('-------- Solve Det --------------------')
    x_seq_det,obj_det,time_det = det.det_seq(n,r_mu,p_mu_esti)
    tft_det = out_sample.computeTotal_det_release(n,test_data,r_mu,S_test,x_seq_det)
    sol = {}
    sol['obj'] = obj_det
    sol['seq'] = x_seq_det
    sol['time'] = time_det
    sol['out_obj'] = tft_det

    print('Det time = ',np.round(time_det,2),\
          ',mean =',np.round(np.mean(tft_det),2),\
            ',quantile 95 =',np.round(np.quantile(tft_det,0.95),2))

    with open(full_path+'sol_det.pkl', "wb") as tf:
        pickle.dump(sol,tf)
    return sol