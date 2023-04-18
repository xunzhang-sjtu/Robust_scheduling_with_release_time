import numpy as np
import pathlib
import pickle

def set_para():
    para = {}
    para['delta_mu'] = 4 # control lb of mean processing time
    para['delta_r'] = 0.05 # control ub of the release time
    para['delta_ep'] = 2 # control the upper bound of the mad
    para['delta_er'] = 1 # control the upper bound of the mad
    para['S_train'] = 20
    para['S_test'] = 10000
    para['iterations'] = 1
    para['instances'] = 10
    para['range_c'] = np.insert(np.arange(0.1,0.50001,0.05),0,np.arange(0,0.1,0.02))
    # para['range_c'] = np.arange(0,0.06,0.05)
    para['range_c'][0] = 1e-6
    para['range_c'] = para['range_c'] + 1

    return para

def get_para(para,name,value,full_path):
    para[name] = value

    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    with open(full_path+'para_info.pkl', "wb") as tf:
        pickle.dump(para,tf)
    return para