import numpy as np
def set_para():
    para = {}
    para['delta_mu'] = 4 # control lb of mean processing time
    para['delta_r'] = 0.1 # control ub of the release time
    para['delta_ep'] = 2 # control the upper bound of the mad
    para['S_train'] = 20
    para['S_test'] = 10000
    para['iterations'] = 1
    para['instances'] = 20
    para['range_c'] = np.arange(1e-6,0.2001,0.04) + 1

    return para

def get_para(para,name):
    return para[name]