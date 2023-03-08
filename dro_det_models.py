
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 20:18:59 2022

@author: xunzhang
"""
import numpy as np

import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum

from rsome import ro                                    # import the ro module
from rsome import norm                                  # import the norm function
from rsome import grb_solver as grb                     # import the Gurobi interface
from numpy import inf
from rsome import dro
from rsome import E
import sche_vns_det_release as sv
import time

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
    # model.setParam('MIPGap',0.01)
    model.setParam('TimeLimit',3600)
    # model.setParam('ConcurrentMIP',6)

    # model.write("dro_e.LP")
   
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
        x_seq = np.zeros(N)   

    return obj_val,x_seq,cpu_time



def det_release_time_scheduling_wass(N,r,c,M,p_hat,d_bar,d_low):


    model = gp.Model('wass')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')
    lbd = model.addVars(M,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    # lbd = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')

    y = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'y')

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



    M_val = 10000000
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
    model.setParam('TimeLimit',3600)
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
    sol['c'] = c
    sol['time'] = cpu_time
    sol['x_seq'] = x_seq
    sol['obj'] = obj_val

    # print('kappa',ka.x)
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
    model.setParam('TimeLimit',3600)
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


def det_release_time_scheduling_wass_without_x(N,r,c,M,p_hat,d_bar,d_low):


    model = gp.Model('wass')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')
    lbd = model.addVars(M,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    # lbd = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')

    model.setObjective(c*ka + (1/M)*quicksum(theta),GRB.MINIMIZE)

    lhs = {}
    v = {}
    # a = {}
    # b = {}
    for m in range(M):
        v[m] = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'v_'+str(m))
        # a[m] = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'a_'+str(m))
        # b[m] = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = 0,name = 'b_'+str(m))

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
                    model.addConstr(v[m][i,j] >= -r[i] * (j-i) \
                        - ka*(d_bar[N-1]-p_hat[N-1,m]) \
                        + d_bar[N-1] ) 
                    model.addConstr(v[m][i,j] >= -r[i] * (j-i) \
                        + p_hat[N-1,m] ) 
                else:
                    model.addConstr(v[m][i,j] >= -(r[i]-r[i-1]) * (j-i)  \
                        - ka * (d_bar[i-1]-p_hat[i-1,m]) \
                        + (j-i+1)*d_bar[i-1]) 
                    model.addConstr(v[m][i,j] >= -(r[i]-r[i-1]) * (j-i) \
                        + p_hat[i-1,m] ) 

        # for i in range(N):
        #     for j in range(i,N+1):
        #         if i == 0:
        #             model.addConstr(v[m][i,j] >= -r[i] * (j-i) + (j-i+1)*p_hat[N-1,m] \
        #                 + a[m][N-1,j]* (d_low[N-1]-p_hat[N-1,m]) + b[m][N-1,j]*(d_bar[N-1] - p_hat[N-1,m])) 
        #             model.addConstr(ka >= j-i+1 - a[m][N-1,j] - b[m][N-1,j]) 
        #             model.addConstr(ka >= -(j-i+1 - a[m][N-1,j] - b[m][N-1,j])) 
        #         else:
        #             model.addConstr(v[m][i,j] >= -(r[i]-r[i-1]) * (j-i) + (j-i+1)*p_hat[i-1,m] \
        #                 + a[m][i-1,j]* (d_low[i-1]-p_hat[i-1,m]) + b[m][i-1,j]*(d_bar[i-1] - p_hat[i-1,m])) 
        #             model.addConstr(ka >= j-i+1 - a[m][i-1,j] - b[m][i-1,j]) 
        #             model.addConstr(ka >= -(j-i+1 - a[m][i-1,j] - b[m][i-1,j])) 


    model.setParam('OutputFlag', 0)
    # model.setParam('MIPGap',0.05)
    model.setParam('TimeLimit',3600)
    # model.setParam('ConcurrentMIP',6)

    model.write("dro.LP")
    start_time = time.time()
    model.optimize()    
    end_time = time.time()
    cpu_time = end_time - start_time
    if model.status == 2 or model.status == 13:
        obj_val = model.getObjective().getValue()



    else:
        obj_val = -1000

    sol = {}

    sol['obj'] = obj_val

    # print('kappa',ka.x)
    return sol

def det_release_time_scheduling_wass_affine_rsome(N,r,c,M,p_hat,d_bar,d_low):


    model = ro.Model('affine')
    ka = model.dvar()
    theta = model.dvar(M)
    p = model.rvar(N)
    # z = model.rvar(1)
    t = model.ldr(N)
    t.adapt(p)

    model.min(c*ka + (1/M)*theta.sum())

    model.st(ka >= 0)
 
    dual_norm1 = {}
    for m in range(M):
        dual_norm1[m] = model.dvar(N)
        zset = (p >= d_low,p <= d_bar)
        model.st((theta[m] >= t.sum() + p.sum() - dual_norm1[m] @ (p-p_hat[:,m]) ).forall(zset))
        model.st( norm(dual_norm1[m],np.inf) <= ka)

        for i in range(N):
            model.st((t[i] >= r[i]).forall(zset))

        for i in range(1,N):
            model.st((t[i]-t[i-1] >= p[i]).forall(zset))

    model.solve(grb)


    sol = {}
    # sol['c'] = c
    # sol['obj'] = obj_val
    # sol['x_seq'] = x_seq
    # sol['time'] = cpu_time
    return sol

def det_release_time_scheduling_wass_affine(N,c,M,p_hat,p_low,p_bar,r):


    model = gp.Model('affine')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')

    t0 = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't0')
    t1 = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 't1')

    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')

    model.setObjective(c*ka + (1/M)*quicksum(theta),GRB.MINIMIZE)

    phi = {}
    bet = {}
    up = {}
    vp = {}

    wp = {}
    sp = {}
    phi_p = {}
    pi_p = {}
    for m in range(M):
        phi[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'alp')
        bet[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'bet')
        up[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0,name = 'up')
        vp[m] = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'sp')


        model.addConstr(theta[m] >= quicksum(t0) + quicksum([-bet[m][j]*p_hat[j,m] for j in range(N)])\
            + quicksum([up[m][j]*p_low[j] + vp[m][j]*p_bar[j] for j in range(N) ])  )

        for j in range(N):
            model.addConstr(bet[m][j]==up[m][j] + vp[m][j] - quicksum([t1[i,j] for i in range(N)]) - 1)

            model.addConstr(ka >= bet[m][j] )
            model.addConstr(ka >= -bet[m][j])


        wp[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'wp')
        sp[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'vp')
        phi_p[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name = 'phi_p')
        pi_p[m] = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'pi_p')

        for i in range(N):
            model.addConstr(t0[i] >= quicksum([wp[m][i,j]*p_low[j] + sp[m][i,j]*p_bar[j] + x[i,j]*r[j]for j in range(N)]))
            for j in range(N):
                model.addConstr(wp[m][i,j] + sp[m][i,j] == -t1[i,j])


            if i == 0:
                model.addConstr(t0[i] >= quicksum([ phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                for j in range(N):
                    model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == -t1[i,j])    

            if i > 0:
                model.addConstr(t0[i] - t0[i-1] >= quicksum([ phi_p[m][i,j]*p_low[j] + pi_p[m][i,j]*p_bar[j] for j in range(N)]))
                for j in range(N):
                    model.addConstr(phi_p[m][i,j] + pi_p[m][i,j] == x[i-1,j] + t1[i-1,j]-t1[i,j])  



    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    model.setParam('OutputFlag', 0)
    # model.setParam('MIPGap',0.05)
    model.setParam('TimeLimit',3600)
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
    sol['c'] = c
    sol['obj'] = obj_val
    sol['x_seq'] = x_seq
    sol['time'] = cpu_time
    return sol