# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:26:55 2020

@author: xunzhang
"""

import pathlib
import numpy as np
import pandas as pd
import computation.compute_det as det
import computation.compute_saa as saa
import computation.compute_wass as wass
import computation.compute_mom as mom
import multiprocessing as mp
import os
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import mosek_models
import pickle
import data_generation as dg


from mosek.fusion import *
import sys


def obtain_sub_model(M,N):

    # with Model() as model:
    model = Model()
    # Create variable 'x' of length 4
    theta = model.variable("theta",M, Domain.unbounded())
    lbd = model.variable("lbd", [M,N], Domain.unbounded())
    # ka = model.variable("ka",1,Domain.greaterThan(0))

    xd = model.parameter("xd",N)
    xr = model.parameter("xr",N)
    xp = model.parameter("xp",[N,M])
    ka = model.parameter("ka")
    c = model.parameter("c")

    v = {}
    for m in range(M):
        v[m] = model.variable([N,N+1],Domain.unbounded())
        model.constraint(Expr.sub(theta.index(m),Expr.sum(lbd.pick([[m,i] for i in range(N)]))), Domain.greaterThan(0.0))

        for k in range(N):
            for j in range(k,N+1,1):
                if j < N:
                    model.constraint(Expr.sub(Expr.sum(lbd.pick([[m,ind] for ind in range(k,j+1)])),\
                                                Expr.sum(v[m].pick([[ind,j] for ind in range(k,j+1)])),),\
                                            Domain.greaterThan(0) )
                else:
                    model.constraint(Expr.sub(Expr.sum(lbd.pick([[m,ind] for ind in range(k,j)])),\
                                                Expr.sum(v[m].pick([[ind,j] for ind in range(k,j)])),),\
                                            Domain.greaterThan(0) )

        for i in range(N):
            for j in range(i,N+1):
                if i == 0:
                    const_name1 = 'm_'+str(m)+'_i_'+str(i)+'_j_'+str(j)+'_v1'
                    const_name2 = 'm_'+str(m)+'_i_'+str(i)+'_j_'+str(j)+'_v2'
                    model.constraint(const_name1,Expr.sub(v[m].pick([[i,j]]),Expr.add(Expr.mul(i-j,xr.index(0)),xp.slice([N-1,m],[N,m+1]))),Domain.greaterThan(0.0))
                    model.constraint(const_name2,Expr.sub(v[m].pick([[i,j]]),Expr.sub(Expr.add(Expr.mul(i-j,xr.index(0)), xp.slice([N-1,m],[N,m+1])), \
                                                                          Expr.sub(Expr.mul(ka,xd.index(N-1)),Expr.mul(ka,xp.slice([N-1,m],[N,m+1]))))),Domain.greaterThan(0))

                else:
                    const_name1 = 'm_'+str(m)+'_i_'+str(i)+'_j_'+str(j)+'_v1'
                    const_name2 = 'm_'+str(m)+'_i_'+str(i)+'_j_'+str(j)+'_v2'
                    model.constraint(const_name1,Expr.sub(v[m].pick([[i,j]]),\
                                                Expr.add(Expr.mul(i-j,Expr.sub(xr.index(i),xr.index(i-1))), \
                                                        Expr.mul(j-i+1,xp.slice([i-1,m],[i,m+1]))) ),Domain.greaterThan(0))
                    model.constraint(const_name2,Expr.sub(v[m].pick([[i,j]]),\
                                                Expr.sub(Expr.add(Expr.mul(i-j,Expr.sub(xr.index(i),xr.index(i-1))), \
                                                                Expr.mul(j-i+1,xd.index([i-1]))), \
                                                                Expr.sub(Expr.mul(ka,xd.index(i-1)),Expr.mul(ka,xp.slice([i-1,m],[i,m+1]))))),Domain.greaterThan(0))

    model.objective("obj", ObjectiveSense.Minimize, Expr.add(Expr.mul(c,ka),Expr.mul(1/M,Expr.sum(theta))))


    return model


def obtain_master_model(M,N):
    model = Model()
    x = model.variable("x",[N,N], Domain.binary())
    # x = model.variable("x",[N,N], Domain.greaterThan(0))

    theta = model.variable('theta',M,Domain.greaterThan(-10000000))

    for j in range(N):
        model.constraint(Expr.sum(x.pick([[j,i] for i in range(N)])),Domain.equalsTo(1))
        model.constraint(Expr.sum(x.pick([[i,j] for i in range(N)])),Domain.equalsTo(1))

    model.objective("obj", ObjectiveSense.Minimize,Expr.sum(theta))
    return model



def solve_master_model_directly(M,N,dual1_all,dual2_all,r_mu,p_bar,train_data,ka,c):
    model = Model()
    x = model.variable("x",[N,N], Domain.binary())
    # x = model.variable("x",[N,N], Domain.greaterThan(0))

    theta = model.variable('theta',M,Domain.greaterThan(-10000000))

    xr = Expr.mul(x,r_mu)
    xd = Expr.mul(x,p_bar)
    xp = Expr.mul(x,train_data)
    Q_appro = {}
    N = n

    for ind in range(len(dual1_all)):
        dual1_set = dual1_all[ind]
        dual2_set = dual2_all[ind]
        for m in range(S_train):
            # Q_appro[m] = Expr.mul(1,theta.index(m))
            for i in range(n):
                for j in range(i,n+1):
                    dual1 = dual1_set[m,i,j]
                    dual2 = dual2_set[m,i,j]
                    if i == 0:
                        Q_appro[ind,m] = Expr.mul(dual1,Expr.add(Expr.mul(i-j,xr.index(0)),xp.slice([N-1,m],[N,m+1])))
                        Q_appro[ind,m] = Expr.add(Q_appro[ind,m],Expr.mul(dual2,Expr.sub(Expr.add(Expr.mul(i-j,xr.index(0)), xp.slice([N-1,m],[N,m+1])), \
                                                                            Expr.sub(Expr.mul(ka,xd.index(N-1)),Expr.mul(ka,xp.slice([N-1,m],[N,m+1]))))))

                    else:
                        Q_appro[ind,m] = Expr.add(Q_appro[ind,m],Expr.mul(dual1,Expr.add(Expr.mul(i-j,Expr.sub(xr.index(i-1),xr.index(i))), \
                                                            Expr.mul(j-i+1,xp.slice([i-1,m],[i,m+1])))))
                        
                        Q_appro[ind,m] = Expr.add(Q_appro[ind,m],Expr.mul(dual2,Expr.sub(Expr.add(Expr.mul(i-j,Expr.sub(xr.index(i-1),xr.index(i))), \
                                                                    Expr.mul(j-i+1,xd.index([i-1]))), \
                                                                    Expr.sub(Expr.mul(ka,xd.index(i-1)),Expr.mul(ka,xp.slice([i-1,m],[i,m+1]))))))

        # constr_name = 'it_'+str(it)+'_m'+str(m)
        model.constraint(Expr.sub(theta.index(m),Q_appro[ind,m]),Domain.greaterThan(0))


    for j in range(N):
        model.constraint(Expr.sum(x.pick([[j,i] for i in range(N)])),Domain.equalsTo(1))
        model.constraint(Expr.sum(x.pick([[i,j] for i in range(N)])),Domain.equalsTo(1))

    model.objective("obj", ObjectiveSense.Minimize,Expr.sum(theta))

    model.solve()
    obj = model.primalObjValue() + c * ka
    print('master obj=',obj)
    return model




def solve_sub_model(x_matrix,r_mu,p_bar,train_data,model_sub,c,ka,S_train,n):
                
    xr_tem = x_matrix @ r_mu
    xd_tem = x_matrix @ p_bar
    xp_tem = x_matrix @ train_data

    model_sub.getParameter("xr").setValue(xr_tem)
    model_sub.getParameter("xd").setValue(xd_tem)
    model_sub.getParameter("xp").setValue(xp_tem)
    model_sub.getParameter("c").setValue(c)
    model_sub.getParameter("ka").setValue(ka)
    model_sub.solve()
    print('sub obj=',model_sub.primalObjValue())

    dual1_set = {}
    dual2_set = {}
    for m in range(S_train):
        for i in range(n):
            for j in range(i,n+1):
                const_name1 = 'm_'+str(m)+'_i_'+str(i)+'_j_'+str(j)+'_v1'
                const_name2 = 'm_'+str(m)+'_i_'+str(i)+'_j_'+str(j)+'_v2'
                dual1_set[m,i,j] = model_sub.getConstraint(const_name1).dual()[0]
                dual2_set[m,i,j] = model_sub.getConstraint(const_name2).dual()[0]



    # dual_obj = 0
    # for m in range(S_train):
    #     for i in range(n):
    #         for j in range(i,n+1):
    #             dual1 = dual1_set[m,i,j]
    #             dual2 = dual2_set[m,i,j]
    #             if i == 0:
    #                 dual_obj = dual_obj + dual1 * ( (i-j)*xr_tem[0] + xp_tem[n-1,m]*(j-i+1) )
    #                 dual_obj = dual_obj + dual2 * ((i-j)*xr_tem[0] + xp_tem[n-1,m]*(j-i+1) - ka * xd_tem[n-1] + ka * xp_tem[n-1,m])
    #             else:
    #                 dual_obj = dual_obj + dual1 * ( (i-j)*(xr_tem[i-1] - xr_tem[i]) + xp_tem[i-1,m]*(j-i+1) )
    #                 dual_obj = dual_obj + dual2 * ((i-j)*(xr_tem[i-1] - xr_tem[i]) + xp_tem[i-1,m]*(j-i+1) - ka * xd_tem[i-1] + ka * xp_tem[i-1,m])


    # print('dual obj=',dual_obj + c * ka)



    return dual1_set,dual2_set

def solve_master_model(model_master,r_mu,p_bar,train_data,S_train,n,dual1_set,dual2_set,c,ka,it):
    
    
    x = model_master.getVariable('x')
    theta = model_master.getVariable('theta')
    xr = Expr.mul(x,r_mu)
    xd = Expr.mul(x,p_bar)
    xp = Expr.mul(x,train_data)
    Q_appro = {}
    N = n
    for m in range(S_train):
        # Q_appro[m] = Expr.mul(1,theta.index(m))
        for i in range(n):
            for j in range(i,n+1):
                dual1 = dual1_set[m,i,j]
                dual2 = dual2_set[m,i,j]
                if i == 0:
                    Q_appro[m] = Expr.mul(dual1,Expr.add(Expr.mul(i-j,xr.index(0)),xp.slice([N-1,m],[N,m+1])))
                    Q_appro[m] = Expr.add(Q_appro[m],Expr.mul(dual2,Expr.sub(Expr.add(Expr.mul(i-j,xr.index(0)), xp.slice([N-1,m],[N,m+1])), \
                                                                        Expr.sub(Expr.mul(ka,xd.index(N-1)),Expr.mul(ka,xp.slice([N-1,m],[N,m+1]))))))

                else:
                    Q_appro[m] = Expr.add(Q_appro[m],Expr.mul(dual1,Expr.add(Expr.mul(i-j,Expr.sub(xr.index(i-1),xr.index(i))), \
                                                        Expr.mul(j-i+1,xp.slice([i-1,m],[i,m+1])))))
                    
                    Q_appro[m] = Expr.add(Q_appro[m],Expr.mul(dual2,Expr.sub(Expr.add(Expr.mul(i-j,Expr.sub(xr.index(i-1),xr.index(i))), \
                                                                Expr.mul(j-i+1,xd.index([i-1]))), \
                                                                Expr.sub(Expr.mul(ka,xd.index(i-1)),Expr.mul(ka,xp.slice([i-1,m],[i,m+1]))))))

        constr_name = 'it_'+str(it)+'_m'+str(m)
        model_master.constraint(constr_name,Expr.sub(theta.index(m),Q_appro[m]),Domain.greaterThan(0))
    model_master.solve()
    obj = model_master.primalObjValue() + c * ka
    x_matrix = np.zeros((n,n))
    seq = model_master.getVariable('x').level()

    for i in range(n):
        for j in range(n):
            x_matrix[i,j] = seq[n*i+j]
    return obj,x_matrix,model_master

def main_process(r_mu,mu_p,std_p,n,S_train,S_test,iterations,ins,file_path):
    for it in range(iterations):
        print('----------------------- ins:',ins,' n:',n,' iteration:',it,'-------------------------------------')

        full_path = file_path + 'ins='+str(ins)+'/' + 'n='+str(n)+'/' + 'iteration='+str(it)+'/'
        # if os.path.exists(full_path+'data_info.pkl'):
        if False:
            with open(full_path+'data_info.pkl', "rb") as tf:
                data_info = pickle.load(tf)
            temp = data_info['data']
            p_bar = data_info['p_bar']
            p_low = data_info['p_low']
            train_data = temp[:,0:S_train]
            test_data = temp[:,S_train:S_train+S_test]
        else:
            # temp,p_bar,p_low = generate_LogNormal(mu_p,std_p,n,S_train+S_test,0.1,0.9)
            data_info = dg.generate_Normal(mu_p,std_p,n,S_train+S_test,0.1,0.9)
            temp = data_info['data']
            p_bar = data_info['p_bar']
            p_low = data_info['p_low']
            train_data = temp[:,0:S_train]
            test_data = temp[:,S_train:S_train+S_test]
            # # create a folder to store the data
            # pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
            # with open(full_path+'data_info.pkl', "wb") as tf:
            #     pickle.dump(data_info,tf)

            model_sub = obtain_sub_model(S_train,n)
            model_master = obtain_master_model(S_train,n)
            # x_matrix,x_dict = heuristic.decode()
            
            c = 0.1 * sum(p_bar - p_low)
            ka = n
            import dro_models
            # sol_given_ka = dro_models.det_release_time_scheduling_wass_given_ka(n,r_mu,c,S_train,train_data,p_bar,ka)



            sol = dro_models.det_release_time_scheduling_wass(n,r_mu,c,S_train,train_data,p_bar,p_low,np.eye(n))
            print('exact obj=',sol['obj'],'seq:',sol['x_seq']+1)
            # ka = sol['ka']
            import heuristic
            x_matrix,x_set = heuristic.decode(sol['x_seq']+1)

            x_matrix = np.eye(n)
            it = 0
            dual1_all = {}
            dual2_all = {}
            for i in range(n):
                shaking_neibors = heuristic.shaking(x_matrix @ np.arange(1,1+n))
                for j in range(2):
                    x_matrix,x_set = heuristic.decode(shaking_neibors[j])
                    dual1_set,dual2_set = solve_sub_model(x_matrix,r_mu,p_bar,train_data,model_sub,c,ka,S_train,n)
                    obj,x_matrix,model_master = solve_master_model(model_master,r_mu,p_bar,train_data,S_train,n,dual1_set,dual2_set,c,ka,it)
                    
                    print('master obj=',obj,'seq:',x_matrix @ np.arange(1,1+n))
                    print('-----------------------')
                    dual1_all[it] = dual1_set
                    dual2_all[it] = dual2_set
                    it = it + 1
            
                solve_master_model_directly(S_train,n,dual1_all,dual2_all,r_mu,p_bar,train_data,ka,c)



    # dual1_set,dual2_set = solve_sub_model(x_matrix,r_mu,p_bar,train_data,model_sub,c,ka,S_train,n)


        # if n <= 20:
        #     exact_model = True
        #     sol_wass_exact = wass.wass_DRO(n,r_mu,train_data,test_data,p_bar,p_low,sol_saa,exact_model,range_c,full_path,model_DRO,models_DRO)




project_path = '/Users/zhangxun/data/robust_scheduling/det_release/vns_vs_exact/'
delta_mu = 4 # control lb of mean processing time
delta_r = 0.1 # control ub of the release time
delta_ep = 1 # control the upper bound of the mad
S_train = 10
S_test = 10000
iterations = 1
instances = 1
range_c = np.arange(0,1.001,0.2)
if __name__ == '__main__':

    n = 3
    ins = 1
    file_path = project_path

    mu_p = np.random.uniform(10*delta_mu,50,n)
    r_mu = np.round(np.random.uniform(0,delta_r*mu_p.sum(),n))
    mad_p = np.random.uniform(0,delta_ep*mu_p)
    std_p = np.sqrt(np.pi/2)*mad_p

    main_process(r_mu,mu_p,std_p,n,S_train,S_test,iterations,ins,file_path)



 
 




