
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 20:18:59 2022

@author: xunzhang
"""
import numpy as np
import copy
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum

from rsome import ro                                    # import the ro module
from rsome import norm                                  # import the norm function
from rsome import grb_solver as grb                     # import the Gurobi interface
from numpy import inf
from rsome import dro
from rsome import E
import time
time_limits = 600
mip_gap = 0.01

def det_release_time_scheduling_moments(N,mu,r,p_bar,p_low):
    # mu = np.round(mu,1)
    # sigma = np.round(sigma,1)
    # r = np.round(r,1)


    model = gp.Model('mad')
    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')
    eta = model.addVars(N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'eta')



    obj = quicksum(eta)
    for i in range(N-1):
        for j in range(N):
            obj = obj + (N-i-1)*mu[j]*x[i,j]
    model.setObjective(obj,GRB.MINIMIZE)

    for k in range(N+1):
        max_index = min(k,N)
        rhs = 0
        for i in range(1,max_index):
            rhs = rhs +  quicksum([(x[i-1,j] - x[i,j]) * r[j] * (k-i) for j in range(N)]) 
        model.addConstr(quicksum([eta[i] for i in range(k+1)]) >= rhs)

    for l in range(1,N+1):
        for k in range(l,N+1):
            max_index = min(k,N)
            rhs = 0
            for i in range(l,max_index):
                rhs = rhs +  quicksum([(x[i-1,j] - x[i,j]) * r[j] * (k-i) for j in range(N)]) 
            model.addConstr(quicksum([eta[i] for i in range(l,k+1)]) >= rhs)

    
    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)



    model.setParam('OutputFlag', 0)
    model.setParam('MIPGap',mip_gap)
    model.setParam('TimeLimit',time_limits)
    # model.setParam('ConcurrentMIP',6)

    # model.write("dro_e.LP")
   
    start_time = time.time()
    model.optimize()    
    end_time = time.time()
    cpu_time = end_time - start_time

    if model.status == 2 or model.status == 13 or model.status == 9:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x
        x_seq = x_tem @ np.arange(N)
    else:
        obj_val = -1000
        x_seq = np.zeros(N)   

    return obj_val,x_seq,cpu_time



def det_release_time_scheduling_wass(N,r,c,M,p_hat,d_bar,d_low,x_saa):


    model = gp.Model('wass')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')
    lbd = model.addVars(M,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    # lbd = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')

    y = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'y')

    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')
    # x = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0, name = 'x')



    model.setObjective(c*ka + (1/M)*quicksum(theta),GRB.MINIMIZE)

    lhs = {}
    v = {}
    for m in range(M):
        v[m] = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'v')

        model.addConstr(theta[m] >= quicksum([lbd[m,j] for j in range(N)]))

        for k in range(N):
            for j in range(k,N+1,1):
                max_index = min(j,N-1)
                lhs[m,k,j] = 0
                for i in range(k,max_index+1):
                    lhs[m,k,j] = lhs[m,k,j] + lbd[m,i] - v[m][i,j]
                model.addConstr(lhs[m,k,j] >= 0)


        for i in range(N):
            for j in range(i,N+1):
                if i == 0:
                    model.addConstr(v[m][i,j] >= -quicksum([x[i,q] * r[q] for q in range(N)]) * (j-i) \
                        - quicksum([y[N-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([x[N-1,q]*d_bar[q] for q in range(N)]) ) 
                    model.addConstr(v[m][i,j] >= -quicksum([x[i,q] * r[q] for q in range(N)]) * (j-i) \
                        + quicksum([x[N-1,q]*p_hat[q,m] for q in range(N)]) ) 
                else:
                    model.addConstr(v[m][i,j] >= -quicksum([(x[i,q] - x[i-1,q]) * r[q] * (j-i) for q in range(N)]) \
                        - quicksum([y[i-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([(j-i+1)*x[i-1,q]*d_bar[q] for q in range(N)]) )
                    model.addConstr(v[m][i,j] >= -quicksum([(x[i,q] - x[i-1,q]) * r[q] * (j-i) for q in range(N)]) \
                        + quicksum([(j-i+1)*x[i-1,q]*p_hat[q,m] for q in range(N)]) ) 



    M_val = N + 0.0001
    for i in range(N):
        for j in range(N):
            model.addConstr(y[i,j] <= ka)
            model.addConstr(y[i,j] <= M_val * x[i,j])
            model.addConstr(y[i,j] >= ka- M_val * (1-x[i,j]))

            # x[i,j].start = x_saa[i,j]


    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    model.setParam('OutputFlag', 1)
    model.setParam('MIPGap',mip_gap)
    model.setParam('TimeLimit',time_limits)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
    start_time = time.time()
    model.optimize()    
    end_time = time.time()
    cpu_time = end_time - start_time
    # print('wass dro run time=',cpu_time)
    if model.status == 2 or model.status == 13 or model.status == 9:
    # if True:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x

        y_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                y_tem[i,j] = y[i,j].x
        x_seq = x_tem @ np.arange(N)

    else:
        obj_val = -1000
        x_seq = np.arange(N)

    sol = {}
    sol['c'] = c
    sol['time'] = cpu_time
    sol['x_seq'] = x_seq
    sol['obj'] = obj_val
    sol['ka'] = ka.x
    return sol


# ---------
def det_release_time_scheduling_wass_given_ka(N,r,c,M,p_hat,d_bar,ka):


    model = gp.Model('wass')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')
    lbd = model.addVars(M,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    # lbd = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')

    model.setObjective(c*ka + (1/M)*quicksum(theta),GRB.MINIMIZE)

    lhs = {}
    v = {}
    for m in range(M):
        v[m] = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'v')

        model.addConstr(theta[m] >= quicksum([lbd[m,j] for j in range(N)]))

        for k in range(N):
            for j in range(k,N+1,1):
                max_index = min(j,N-1)
                lhs[m,k,j] = 0
                for i in range(k,max_index+1):
                    lhs[m,k,j] = lhs[m,k,j] + lbd[m,i] - v[m][i,j]
                model.addConstr(lhs[m,k,j] >= 0)

        # for k in range(N):
        #     for j in range(k,N+1,1):
        #         if j < N:
        #             model.addConstr(quicksum([lbd[m,ind] for ind in range(k,j+1)])-\
        #                             quicksum([v[m][ind,j] for ind in range(k,j+1)]) >=0 )
        #         else:

        #             model.addConstr(quicksum([lbd[m,ind] for ind in range(k,j)])-\
        #                          quicksum([v[m][ind,j] for ind in range(k,j)]) >=0 )




        for i in range(N):
            for j in range(i,N+1):
            # for j in [0,N-1,N]:
                if i == 0:
                    model.addConstr(v[m][i,j] >= -quicksum([x[i,q] * r[q] for q in range(N)]) * (j-i) \
                        - quicksum([ka*x[N-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([x[N-1,q]*d_bar[q] for q in range(N)]) ) 
                    model.addConstr(v[m][i,j] >= -quicksum([x[i,q] * r[q] for q in range(N)]) * (j-i) \
                        + quicksum([x[N-1,q]*p_hat[q,m] for q in range(N)]) ) 
                else:
                    model.addConstr(v[m][i,j] >= -quicksum([(x[i,q] - x[i-1,q]) * r[q] * (j-i) for q in range(N)]) \
                        - quicksum([ka*x[i-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([(j-i+1)*x[i-1,q]*d_bar[q] for q in range(N)]) )
                    model.addConstr(v[m][i,j] >= -quicksum([(x[i,q] - x[i-1,q]) * r[q] * (j-i) for q in range(N)]) \
                        + quicksum([(j-i+1)*x[i-1,q]*p_hat[q,m] for q in range(N)]) ) 


    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    model.setParam('OutputFlag', 0)
    # model.setParam('MIPGap',0.01)
    model.setParam('TimeLimit',time_limits)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
    start_time = time.time()
    model.optimize()    
    end_time = time.time()
    cpu_time = end_time - start_time
    if model.status == 2 or model.status == 13 or model.status == 9:
    # if True:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x

        x_seq = x_tem @ np.arange(N)

    else:
        obj_val = -1000
        x_seq = np.arange(N)

    sol = {}
    sol['c'] = c
    sol['time'] = cpu_time
    sol['x_seq'] = x_seq
    sol['obj'] = obj_val

    return sol

def det_release_time_scheduling_RS(N,r,tau,M,p_hat,d_bar,d_low):


    model = gp.Model('wass')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')
    lbd = model.addVars(M,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    y = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'y')

    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')

    model.setObjective(ka,GRB.MINIMIZE)

    model.addConstr((1/M)*quicksum(theta)<= tau)

    lhs = {}
    v = {}
    for m in range(M):
        v[m] = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'v')

        model.addConstr(theta[m] >= quicksum([lbd[m,j] for j in range(N)]))

        for k in range(N):
            for j in range(k,N+1,1):
                max_index = min(j,N-1)
                lhs[m,k,j] = 0
                for i in range(k,max_index+1):
                    lhs[m,k,j] = lhs[m,k,j] + lbd[m,i] - v[m][i,j]
                model.addConstr(lhs[m,k,j] >= 0)


        for i in range(N):
            for j in range(i,N+1):
                if i == 0:
                    model.addConstr(v[m][i,j] >= -quicksum([x[i,q] * r[q] for q in range(N)]) * (j-i) \
                        - quicksum([y[N-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([x[N-1,q]*d_bar[q] for q in range(N)]) ) 
                    model.addConstr(v[m][i,j] >= -quicksum([x[i,q] * r[q] for q in range(N)]) * (j-i) \
                        + quicksum([x[N-1,q]*p_hat[q,m] for q in range(N)]) ) 
                else:
                    model.addConstr(v[m][i,j] >= -quicksum([(x[i,q] - x[i-1,q]) * r[q] * (j-i) for q in range(N)]) \
                        - quicksum([y[i-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([(j-i+1)*x[i-1,q]*d_bar[q] for q in range(N)]) )
                    model.addConstr(v[m][i,j] >= -quicksum([(x[i,q] - x[i-1,q]) * r[q] * (j-i) for q in range(N)]) \
                        + quicksum([(j-i+1)*x[i-1,q]*p_hat[q,m] for q in range(N)]) ) 

    M_val = N+1
    for i in range(N):
        for j in range(N):
            model.addConstr(y[i,j] <= ka)
            model.addConstr(y[i,j] <= M_val * x[i,j])
            model.addConstr(y[i,j] >= ka- M_val * (1-x[i,j]))

    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    model.setParam('OutputFlag', 0)
    # model.setParam('MIPGap',0.05)
    model.setParam('TimeLimit',time_limits)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    start_time = time.time()
    model.optimize()    
    end_time = time.time()
    cpu_time = end_time - start_time
    if model.status == 2 or model.status == 13:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x

        y_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                y_tem[i,j] = y[i,j].x
        x_seq = x_tem @ np.arange(N)

    else:
        obj_val = -1000
        x_seq = np.arange(N)
    sol = {}
    sol['c'] = tau
    sol['time'] = cpu_time
    sol['x_seq'] = x_seq
    sol['obj'] = obj_val

    # print('kappa',ka.x)
    return sol


# **********************************

def det_release_time_scheduling_RS_given_ka(N,r,M,p_hat,d_bar,ka):


    model = gp.Model('wass')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')
    lbd = model.addVars(M,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')

    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')

    model.setObjective((1/M)*quicksum(theta),GRB.MINIMIZE)

    lhs = {}
    v = {}
    for m in range(M):
        v[m] = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'v')

        model.addConstr(theta[m] >= quicksum([lbd[m,j] for j in range(N)]))

        for k in range(N):
            for j in range(k,N+1,1):
                max_index = min(j,N-1)
                lhs[m,k,j] = 0
                for i in range(k,max_index+1):
                    lhs[m,k,j] = lhs[m,k,j] + lbd[m,i] - v[m][i,j]
                model.addConstr(lhs[m,k,j] >= 0)


        for i in range(N):
            for j in range(i,N+1):
                if i == 0:
                    model.addConstr(v[m][i,j] >= -quicksum([x[i,q] * r[q] for q in range(N)]) * (j-i) \
                        - quicksum([ka*x[N-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([x[N-1,q]*d_bar[q] for q in range(N)]) ) 
                    model.addConstr(v[m][i,j] >= -quicksum([x[i,q] * r[q] for q in range(N)]) * (j-i) \
                        + quicksum([x[N-1,q]*p_hat[q,m] for q in range(N)]) ) 
                else:
                    model.addConstr(v[m][i,j] >= -quicksum([(x[i,q] - x[i-1,q]) * r[q] * (j-i) for q in range(N)]) \
                        - quicksum([ka*x[i-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([(j-i+1)*x[i-1,q]*d_bar[q] for q in range(N)]) )
                    model.addConstr(v[m][i,j] >= -quicksum([(x[i,q] - x[i-1,q]) * r[q] * (j-i) for q in range(N)]) \
                        + quicksum([(j-i+1)*x[i-1,q]*p_hat[q,m] for q in range(N)]) ) 

    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    model.setParam('OutputFlag', 0)
    # model.setParam('MIPGap',0.05)
    model.setParam('TimeLimit',time_limits)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    start_time = time.time()
    model.optimize()    
    end_time = time.time()
    cpu_time = end_time - start_time
    if model.status == 2 or model.status == 13:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x
        x_seq = x_tem @ np.arange(N)

    else:
        obj_val = -1000
        x_seq = np.arange(N)
    sol = {}
    sol['time'] = cpu_time
    sol['x_seq'] = x_seq
    sol['obj'] = obj_val

    # print('kappa',ka.x)
    return sol

def det_release_time_scheduling_wass_affine(N,c,M,r,p_hat,p_low,p_bar):


    model = gp.Model('affine')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')

    t0 = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't0')
    t1 = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't1')

    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')

    model.setObjective(c*ka + (1/M)*quicksum(theta),GRB.MINIMIZE)

    bet = {}
    up = {}
    vp = {}

    wp = {}
    sp = {}

    phi_p = {}
    pi_p = {}
    for m in range(M):
        bet[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'bet')
        up[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0,name = 'up')
        vp[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'sp')


        model.addConstr(theta[m] >= quicksum(t0) + quicksum([-bet[m][j]*p_hat[j,m] for j in range(N)])\
            + quicksum([up[m][j]*p_low[j] + vp[m][j]*p_bar[j] for j in range(N) ]))

        for j in range(N):
            model.addConstr(bet[m][j]==up[m][j] + vp[m][j] - quicksum([t1[i,j] for i in range(N)]) - 1)

            model.addConstr(ka >= bet[m][j])
            model.addConstr(ka >= -bet[m][j])


        wp[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'wp')
        sp[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'vp')

        phi_p[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'phi_p')
        pi_p[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'pi_p')

        for i in range(N):
            model.addConstr(t0[i] - quicksum([x[i,j] * r[j] for j in range(N)]) >= quicksum([ wp[m][i,j]*p_low[j] + sp[m][i,j]*p_bar[j] for j in range(N)]))
            for j in range(N):
                model.addConstr(wp[m][i,j] + sp[m][i,j] == -t1[i,j])


            if i == 0:
                model.addConstr(t0[i] >= quicksum([phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                for j in range(N):
                    model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == -t1[i,j])    

            if i > 0:
                model.addConstr(t0[i] - t0[i-1] >= quicksum([phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                for j in range(N):
                    model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == x[i-1,j] + t1[i-1,j]-t1[i,j])  



    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    model.setParam('OutputFlag', 1)
    model.setParam('MIPGap',mip_gap)
    model.setParam('TimeLimit',time_limits)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    cpu_time = end_time - start_time   
    if model.status == 2 or model.status == 13 or model.status == 9:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x

        x_seq = x_tem @ np.arange(N)

    else:
        obj_val = np.NAN
        x_seq = np.ones(N)*np.NAN

    sol = {}
    sol['c'] = c
    sol['obj'] = obj_val
    sol['x_seq'] = x_seq
    sol['time'] = cpu_time
    return sol

# --- det affine given ka -----
def det_release_time_scheduling_wass_affine_given_ka(N,c,M,r,p_hat,p_low,p_bar,ka):

    model = gp.Model('affine')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')
    t0 = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't0')
    t1 = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't1')
    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')
    model.setObjective(c*ka + (1/M)*quicksum(theta),GRB.MINIMIZE)

    bet = {}
    up = {}
    vp = {}

    wp = {}
    sp = {}

    phi_p = {}
    pi_p = {}
    for m in range(M):
        bet[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'bet')
        up[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0,name = 'up')
        vp[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'sp')


        model.addConstr(theta[m] >= quicksum(t0) + quicksum([-bet[m][j]*p_hat[j,m] for j in range(N)])\
            + quicksum([up[m][j]*p_low[j] + vp[m][j]*p_bar[j] for j in range(N) ]))

        for j in range(N):
            model.addConstr(bet[m][j]==up[m][j] + vp[m][j] - quicksum([t1[i,j] for i in range(N)]) - 1)
            model.addConstr(ka >= bet[m][j])
            model.addConstr(ka >= -bet[m][j])

        wp[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'wp')
        sp[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'vp')

        phi_p[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'phi_p')
        pi_p[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'pi_p')

        for i in range(N):
            model.addConstr(t0[i] - quicksum([x[i,j] * r[j] for j in range(N)]) >= quicksum([ wp[m][i,j]*p_low[j] + sp[m][i,j]*p_bar[j] for j in range(N)]))
            for j in range(N):
                model.addConstr(wp[m][i,j] + sp[m][i,j] == -t1[i,j])


            if i == 0:
                model.addConstr(t0[i] >= quicksum([phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                for j in range(N):
                    model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == -t1[i,j])    

            if i > 0:
                model.addConstr(t0[i] - t0[i-1] >= quicksum([phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                for j in range(N):
                    model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == x[i-1,j] + t1[i-1,j]-t1[i,j])  



    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    model.setParam('OutputFlag', 1)
    model.setParam('MIPGap',mip_gap)
    model.setParam('TimeLimit',time_limits)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    cpu_time = end_time - start_time   
    if model.status == 2 or model.status == 13 or model.status == 9:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x

        x_seq = x_tem @ np.arange(N)

    else:
        obj_val = np.NAN
        x_seq = np.ones(N)*np.NAN

    sol = {}
    sol['c'] = c
    sol['obj'] = obj_val
    sol['x_seq'] = x_seq
    sol['time'] = cpu_time
    return sol


def rand_release_time_scheduling_wass(N,c,M,r_hat,p_hat,d_bar,r_low,r_bar):


    model = gp.Model('wass')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')
    lbd = model.addVars(M,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    y = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'y')

    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')

    model.setObjective(c*ka + (1/M)*quicksum(theta),GRB.MINIMIZE)

    lhs = {}
    v = {}
    w = {}
    for m in range(M):
        v[m] = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'v')
        w[m] = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'v')

        model.addConstr(theta[m] >= quicksum([lbd[m,j] for j in range(N)]))

        for k in range(N):
            for j in range(k,N+1,1):
                max_index = min(j,N-1)
                lhs[m,k,j] = 0
                for i in range(k,max_index+1):
                    lhs[m,k,j] = lhs[m,k,j] + lbd[m,i] - v[m][i,j]
                model.addConstr(lhs[m,k,j] >= 0)




        for i in range(N):
            for j in range(i,N+1):
                if i == 0:
                    model.addConstr(v[m][i,j] >= -w[m][i,j] \
                        - quicksum([y[N-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([x[N-1,q]*d_bar[q] for q in range(N)]) ) 
                    model.addConstr(v[m][i,j] >= -w[m][i,j] \
                        + quicksum([x[N-1,q]*p_hat[q,m] for q in range(N)]) )

                    model.addConstr(w[m][i,j] <= (j-i) * quicksum([x[i,q]*r_hat[q,m] for q in range(N)])) 

                    model.addConstr(w[m][i,j] <= (j-i) * quicksum([x[i,q]*r_low[q] for q in range(N)])\
                        - quicksum([y[i,q]*r_low[q] for q in range(N)]) \
                        + quicksum([y[i,q]*r_hat[q,m] for q in range(N)])) 
                else:
                    model.addConstr(v[m][i,j] >= -w[m][i,j] \
                        - quicksum([y[i-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([(j-i+1)*x[i-1,q]*d_bar[q] for q in range(N)]) )
                    model.addConstr(v[m][i,j] >= -w[m][i,j] \
                        + quicksum([(j-i+1)*x[i-1,q]*p_hat[q,m] for q in range(N)]) ) 

                    model.addConstr(w[m][i,j] <= (j-i) * quicksum([(x[i,q]-x[i-1,q])*r_hat[q,m] for q in range(N)])) 

                    model.addConstr(w[m][i,j] <= (j-i) * quicksum([x[i,q]*r_low[q]-x[i-1,q]*r_bar[q] for q in range(N)])\
                        - quicksum([y[i,q]*r_low[q]-y[i-1,q]*r_bar[q] for q in range(N)]) \
                        + quicksum([(y[i,q]-y[i-1,q])*r_hat[q,m] for q in range(N)])) 

    M_val = 1000
    for i in range(N):
        for j in range(N):
            model.addConstr(y[i,j] <= ka)
            model.addConstr(y[i,j] <= M_val * x[i,j])
            model.addConstr(y[i,j] >= ka- M_val * (1-x[i,j]))

    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    model.setParam('OutputFlag', 0)
    # model.setParam('MIPGap',0.05)
    model.setParam('TimeLimit',time_limits)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    cpu_time = end_time - start_time    
    if model.status == 2 or model.status == 13:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x

        y_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                y_tem[i,j] = y[i,j].x
        x_seq = x_tem @ np.arange(N)

    else:
        obj_val = -1000
        x_seq = np.arange(N)
    # print('kappa',ka.x)
    sol = {}
    sol['c'] = c
    sol['obj'] = obj_val
    sol['x_seq'] = x_seq
    sol['time'] = cpu_time
    return sol

def rand_release_time_scheduling_RS(N,tau,M,r_hat,p_hat,d_bar,r_low,r_bar):


    model = gp.Model('wass')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')
    lbd = model.addVars(M,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    y = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'y')

    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')

    model.setObjective(ka,GRB.MINIMIZE)

    model.addConstr((1/M)*quicksum(theta) <= tau)

    lhs = {}
    v = {}
    w = {}
    for m in range(M):
        v[m] = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'v')
        w[m] = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'v')

        model.addConstr(theta[m] >= quicksum([lbd[m,j] for j in range(N)]))

        for k in range(N):
            for j in range(k,N+1,1):
                max_index = min(j,N-1)
                lhs[m,k,j] = 0
                for i in range(k,max_index+1):
                    lhs[m,k,j] = lhs[m,k,j] + lbd[m,i] - v[m][i,j]
                model.addConstr(lhs[m,k,j] >= 0)




        for i in range(N):
            for j in range(i,N+1):
                if i == 0:
                    model.addConstr(v[m][i,j] >= -w[m][i,j] \
                        - quicksum([y[N-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([x[N-1,q]*d_bar[q] for q in range(N)]) ) 
                    model.addConstr(v[m][i,j] >= -w[m][i,j] \
                        + quicksum([x[N-1,q]*p_hat[q,m] for q in range(N)]) )

                    model.addConstr(w[m][i,j] <= (j-i) * quicksum([x[i,q]*r_hat[q,m] for q in range(N)])) 

                    model.addConstr(w[m][i,j] <= (j-i) * quicksum([x[i,q]*r_low[q] for q in range(N)])\
                        - quicksum([y[i,q]*r_low[q] for q in range(N)]) \
                        + quicksum([y[i,q]*r_hat[q,m] for q in range(N)])) 
                else:
                    model.addConstr(v[m][i,j] >= -w[m][i,j] \
                        - quicksum([y[i-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([(j-i+1)*x[i-1,q]*d_bar[q] for q in range(N)]) )
                    model.addConstr(v[m][i,j] >= -w[m][i,j] \
                        + quicksum([(j-i+1)*x[i-1,q]*p_hat[q,m] for q in range(N)]) ) 

                    model.addConstr(w[m][i,j] <= (j-i) * quicksum([(x[i,q]-x[i-1,q])*r_hat[q,m] for q in range(N)])) 

                    model.addConstr(w[m][i,j] <= (j-i) * quicksum([x[i,q]*r_low[q]-x[i-1,q]*r_bar[q] for q in range(N)])\
                        - quicksum([y[i,q]*r_low[q]-y[i-1,q]*r_bar[q] for q in range(N)]) \
                        + quicksum([(y[i,q]-y[i-1,q])*r_hat[q,m] for q in range(N)])) 

    M_val = 1000
    for i in range(N):
        for j in range(N):
            model.addConstr(y[i,j] <= ka)
            model.addConstr(y[i,j] <= M_val * x[i,j])
            model.addConstr(y[i,j] >= ka- M_val * (1-x[i,j]))

    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    model.setParam('OutputFlag', 0)
    # model.setParam('MIPGap',0.05)
    model.setParam('TimeLimit',time_limits)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    cpu_time = end_time - start_time      
    if model.status == 2 or model.status == 13:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x

        y_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                y_tem[i,j] = y[i,j].x
        x_seq = x_tem @ np.arange(N)

    else:
        obj_val = -1000

    # print('kappa',ka.x)
    sol = {}
    sol['c'] = tau
    sol['obj'] = obj_val
    sol['x_seq'] = x_seq
    sol['time'] = cpu_time
    return sol

def rand_release_time_scheduling_wass_affine(N,c,M,r_hat,p_hat,p_low,p_bar,r_low,r_bar):


    model = gp.Model('affine')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')

    t0 = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't0')
    t1 = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't1')
    t2 = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't2')

    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')

    model.setObjective(c*ka + (1/M)*quicksum(theta),GRB.MINIMIZE)

    phi = {}
    bet = {}
    up = {}
    vp = {}
    ur = {}
    vr = {}

    wp = {}
    sp = {}
    wr = {}
    sr = {}
    phi_p = {}
    phi_r = {}
    pi_p = {}
    pi_r = {}
    for m in range(M):
        phi[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'alp')
        bet[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'bet')
        up[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0,name = 'up')
        ur[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'ur')

        vp[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'sp')
        vr[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'sr')


        model.addConstr(theta[m] >= quicksum(t0) + quicksum([-bet[m][j]*p_hat[j,m] - phi[m][j]*r_hat[j,m] for j in range(N)])\
            + quicksum([up[m][j]*p_low[j] + vp[m][j]*p_bar[j] + ur[m][j]*r_low[j] + vr[m][j]*r_bar[j] for j in range(N) ]))

        for j in range(N):
            model.addConstr(bet[m][j]==up[m][j] + vp[m][j] - quicksum([t1[i,j] for i in range(N)]) - 1)
            model.addConstr(phi[m][j]==ur[m][j] + vr[m][j] - quicksum([t2[i,j] for i in range(N)]))

            model.addConstr(ka >= bet[m][j] )
            model.addConstr(ka >= -bet[m][j])

            model.addConstr(ka >= phi[m][j] )
            model.addConstr(ka >= -phi[m][j])




        wp[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'wp')
        wr[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'wr')
        sp[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'vp')
        sr[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'vr')

        phi_p[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'phi_p')
        phi_r[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'phi_r')
        pi_p[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'pi_p')
        pi_r[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'pi_r')

        for i in range(N):
            model.addConstr(t0[i] >= quicksum([wr[m][i,j]*r_low[j] + sr[m][i,j]*r_bar[j] + wp[m][i,j]*p_low[j] + sp[m][i,j]*p_bar[j] for j in range(N)]))
            for j in range(N):
                model.addConstr(wr[m][i,j] + sr[m][i,j] == x[i,j] - t2[i,j])
                model.addConstr(wp[m][i,j] + sp[m][i,j] == -t1[i,j])


            if i == 0:
                model.addConstr(t0[i] >= quicksum([phi_r[m][i,j]*r_low[j] + pi_r[m][i,j]*r_bar[j] + phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                for j in range(N):
                    model.addConstr(phi_r[m][i,j] + pi_r[m][i,j] == -t2[i,j])
                    model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == -t1[i,j])    

            if i > 0:
                model.addConstr(t0[i] - t0[i-1] >= quicksum([phi_r[m][i,j]*r_low[j] + pi_r[m][i,j]*r_bar[j] + phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                for j in range(N):
                    model.addConstr(phi_r[m][i,j] + pi_r[m][i,j] == t2[i-1,j]-t2[i,j])
                    model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == x[i-1,j] + t1[i-1,j]-t1[i,j])  



    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    model.setParam('OutputFlag', 0)
    model.setParam('MIPGap',0.01)
    model.setParam('TimeLimit',time_limits)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    cpu_time = end_time - start_time   
    if model.status == 2 or model.status == 13 or model.status == 9:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x

        x_seq = x_tem @ np.arange(N)

    else:
        obj_val = np.NAN
        x_seq = np.ones(N)*np.NAN

    sol = {}
    sol['c'] = c
    sol['obj'] = obj_val
    sol['x_seq'] = x_seq
    sol['time'] = cpu_time
    return sol


def rand_release_time_scheduling_wass_affine_scenario(N,c,M,r_hat,p_hat,p_low,p_bar,r_low,r_bar):


    model = gp.Model('affine')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')


    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')

    model.setObjective(c*ka + (1/M)*quicksum(theta),GRB.MINIMIZE)

    phi = {}
    bet = {}
    up = {}
    vp = {}
    ur = {}
    vr = {}

    wp = {}
    sp = {}
    wr = {}
    sr = {}
    phi_p = {}
    phi_r = {}
    pi_p = {}
    pi_r = {}

    t0 = {}
    t1 = {}
    t2 = {}
    for m in range(M):

        t0[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't0')
        t1[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't1')
        t2[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't2')


        phi[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'alp')
        bet[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'bet')
        up[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0,name = 'up')
        ur[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'ur')

        vp[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'sp')
        vr[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'sr')


        model.addConstr(theta[m] >= quicksum(t0[m]) + quicksum([-bet[m][j]*p_hat[j,m] - phi[m][j]*r_hat[j,m] for j in range(N)])\
            + quicksum([up[m][j]*p_low[j] + vp[m][j]*p_bar[j] + ur[m][j]*r_low[j] + vr[m][j]*r_bar[j] for j in range(N) ]))

        for j in range(N):
            model.addConstr(bet[m][j]==up[m][j] + vp[m][j] - quicksum([t1[m][i,j] for i in range(N)]) - 1)
            model.addConstr(phi[m][j]==ur[m][j] + vr[m][j] - quicksum([t2[m][i,j] for i in range(N)]))

            model.addConstr(ka >= bet[m][j] )
            model.addConstr(ka >= -bet[m][j])

            model.addConstr(ka >= phi[m][j] )
            model.addConstr(ka >= -phi[m][j])




        wp[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'wp')
        wr[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'wr')
        sp[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'vp')
        sr[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'vr')

        phi_p[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'phi_p')
        phi_r[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'phi_r')
        pi_p[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'pi_p')
        pi_r[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'pi_r')

        for i in range(N):
            model.addConstr(t0[m][i] >= quicksum([wr[m][i,j]*r_low[j] + sr[m][i,j]*r_bar[j] + wp[m][i,j]*p_low[j] + sp[m][i,j]*p_bar[j] for j in range(N)]))
            for j in range(N):
                model.addConstr(wr[m][i,j] + sr[m][i,j] == x[i,j] - t2[m][i,j])
                model.addConstr(wp[m][i,j] + sp[m][i,j] == -t1[m][i,j])


            if i == 0:
                model.addConstr(t0[m][i] >= quicksum([phi_r[m][i,j]*r_low[j] + pi_r[m][i,j]*r_bar[j] + phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                for j in range(N):
                    model.addConstr(phi_r[m][i,j] + pi_r[m][i,j] == -t2[m][i,j])
                    model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == -t1[m][i,j])    

            if i > 0:
                model.addConstr(t0[m][i] - t0[m][i-1] >= quicksum([phi_r[m][i,j]*r_low[j] + pi_r[m][i,j]*r_bar[j] + phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                for j in range(N):
                    model.addConstr(phi_r[m][i,j] + pi_r[m][i,j] == t2[m][i-1,j]-t2[m][i,j])
                    model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == x[i-1,j] + t1[m][i-1,j]-t1[m][i,j])  



    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    model.setParam('OutputFlag', 0)
    # model.setParam('MIPGap',0.05)
    model.setParam('TimeLimit',time_limits)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    cpu_time = end_time - start_time   
    if model.status == 2 or model.status == 13 or model.status == 9:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x

        x_seq = x_tem @ np.arange(N)

    else:
        obj_val = -1000
        x_seq = np.arange(N)
    sol = {}
    sol['c'] = c
    sol['obj'] = obj_val
    sol['x_seq'] = x_seq
    sol['time'] = cpu_time
    return sol



def rand_release_time_scheduling_RS_affine(N,tau,M,r_hat,p_hat,p_low,p_bar,r_low,r_bar):


    model = gp.Model('affine')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')

    t0 = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't0')
    t1 = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't1')
    t2 = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't2')

    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')

    model.setObjective(ka ,GRB.MINIMIZE)
    model.addConstr((1/M)*quicksum(theta)<= tau)

    phi = {}
    bet = {}
    up = {}
    vp = {}
    ur = {}
    vr = {}

    wp = {}
    sp = {}
    wr = {}
    sr = {}
    phi_p = {}
    phi_r = {}
    pi_p = {}
    pi_r = {}
    for m in range(M):
        phi[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'alp')
        bet[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'bet')
        up[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0,name = 'up')
        ur[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'ur')

        vp[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'sp')
        vr[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'sr')


        model.addConstr(theta[m] >= quicksum(t0) + quicksum([-bet[m][j]*p_hat[j,m] - phi[m][j]*r_hat[j,m] for j in range(N)])\
            + quicksum([up[m][j]*p_low[j] + vp[m][j]*p_bar[j] + ur[m][j]*r_low[j] + vr[m][j]*r_bar[j] for j in range(N) ]))

        for j in range(N):
            model.addConstr(bet[m][j]==up[m][j] + vp[m][j] - quicksum([t1[i,j] for i in range(N)]) - 1)
            model.addConstr(phi[m][j]==ur[m][j] + vr[m][j] - quicksum([t2[i,j] for i in range(N)]))

            model.addConstr(ka >= bet[m][j] )
            model.addConstr(ka >= -bet[m][j])

            model.addConstr(ka >= phi[m][j] )
            model.addConstr(ka >= -phi[m][j])




        wp[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'wp')
        wr[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'wr')
        sp[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'vp')
        sr[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'vr')

        phi_p[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'phi_p')
        phi_r[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'phi_r')
        pi_p[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'pi_p')
        pi_r[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'pi_r')

        for i in range(N):
            model.addConstr(t0[i] >= quicksum([wr[m][i,j]*r_low[j] + sr[m][i,j]*r_bar[j] + wp[m][i,j]*p_low[j] + sp[m][i,j]*p_bar[j] for j in range(N)]))
            for j in range(N):
                model.addConstr(wr[m][i,j] + sr[m][i,j] == x[i,j] - t2[i,j])
                model.addConstr(wp[m][i,j] + sp[m][i,j] == -t1[i,j])


            if i == 0:
                model.addConstr(t0[i] >= quicksum([phi_r[m][i,j]*r_low[j] + pi_r[m][i,j]*r_bar[j] + phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                for j in range(N):
                    model.addConstr(phi_r[m][i,j] + pi_r[m][i,j] == -t2[i,j])
                    model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == -t1[i,j])    

            if i > 0:
                model.addConstr(t0[i] - t0[i-1] >= quicksum([phi_r[m][i,j]*r_low[j] + pi_r[m][i,j]*r_bar[j] + phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                for j in range(N):
                    model.addConstr(phi_r[m][i,j] + pi_r[m][i,j] == t2[i-1,j]-t2[i,j])
                    model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == x[i-1,j] + t1[i-1,j]-t1[i,j])  



    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    model.setParam('OutputFlag', 0)
    # model.setParam('MIPGap',0.05)
    model.setParam('TimeLimit',time_limits)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    cpu_time = end_time - start_time   
    if model.status == 2 or model.status == 13:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x

        x_seq = x_tem @ np.arange(N)

    else:
        obj_val = -1000

    sol = {}
    sol['c'] = tau
    sol['obj'] = obj_val
    sol['x_seq'] = x_seq
    sol['time'] = cpu_time
    return sol


def rand_release_time_scheduling_wass_affine_rsome(N,c,M,r_hat,p_hat,p_low,p_bar,r_low,r_bar):


    model = ro.Model('affine')
    ka = model.dvar()
    theta = model.dvar(M)
    p = model.rvar(N)
    r = model.rvar(N)
    # z = model.rvar(1)
    t = model.ldr(N)
    t.adapt(p)
    t.adapt(r)
    # t.adapt(z)
    x = model.dvar((N,N),'B')

    model.minmax(c*ka + (1/M)*theta.sum())

    model.st(ka >= 0)
 
    dual_norm1 = {}
    dual_norm2 = {}
    for m in range(M):
        dual_norm1[m] = model.dvar(N)
        dual_norm2[m] = model.dvar(N)
        zset = (p >= p_low,p <= p_bar,r >= r_low,r<= r_bar)
        model.st((theta[m] >= t.sum() + p.sum() - dual_norm1[m] @ (p-p_hat[:,m]) - dual_norm2[m] @ (r-r_hat[:,m]) ).forall(zset))
        model.st( norm(dual_norm1[m],np.inf) <= ka)
        model.st( norm(dual_norm2[m],np.inf) <= ka)

        for i in range(N):
            model.st((t[i] >= x[i,:]@r).forall(zset))

        for i in range(1,N):
            model.st((t[i]-t[i-1] >= x[i-1,:]@p).forall(zset))

    for i in range(N):
        model.st(x[i,:].sum()==1)
        model.st(x[:,i].sum()==1)

    model.solve(grb)


    sol = {}
    # sol['c'] = c
    # sol['obj'] = obj_val
    # sol['x_seq'] = x_seq
    # sol['time'] = cpu_time
    return sol