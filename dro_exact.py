
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
from rsome import norm
from rsome import E


def det_release_time_scheduling(N,mu,sigma,r,x_given):
    # mu = np.round(mu,1)
    # sigma = np.round(sigma,1)
    # r = np.round(r,1)


    model = gp.Model('det')
    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')
    y = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'y')
    w = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'w')
    xi = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'xi')
    lbd = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    rho1 = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'rho1')
    rho2 = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'rho2')


    obj = 0
    obj = obj + lbd.sum()
    for i in range(N):
        for j in range(N):
            obj = obj + y[i,j] * mu[j] + w[i,j] * sigma[j]
    
    model.setObjective(obj,GRB.MINIMIZE)

    lhs = {}
    for k in range(N):
        for j in range(k,N+1,1):
            max_index = min(j,N-1)
            lhs[k,j] = 0
            for i in range(k,max_index+1):
                # print('k:',k,',j:',j,',i:',i)
                if i == 0:
                    lhs[k,j] = lhs[k,j] + xi[i,j] - lbd[i] - quicksum([x[i,q] * r[q] * (j-i) for q in range(N)])
                else:
                    lhs[k,j] = lhs[k,j] + xi[i,j] - lbd[i] - quicksum([(x[i,q] - x[i-1,q]) * r[q] * (j-i) for q in range(N)])
            model.addConstr(lhs[k,j] <= 0)

    qc = {}
    for i in range(N):
        for j in range(i,N+1):
            name = 'qc['+str(i) + ',' + str(j) + ']'
            qc[i,j] = model.addVars(3,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = name)
            if i == 0:
                model.addConstr(qc[i,j][2] == - rho1[N-1])
                model.addConstr(qc[i,j][1] == rho2[N-1] - xi[i,j])
                model.addConstr(qc[i,j][0] == rho2[N-1] + xi[i,j])
                model.addConstr(qc[i,j][0] >= 0)
                model.addConstr(qc[i,j][2] * qc[i,j][2] +  qc[i,j][1] * qc[i,j][1] <= qc[i,j][0] * qc[i,j][0])
                
                model.addConstr(xi[i,j] >= 0)
            else:
                model.addConstr(qc[i,j][2] == (j-i) - rho1[i-1])
                model.addConstr(qc[i,j][1] == rho2[i-1] - xi[i,j])
                model.addConstr(qc[i,j][0] == rho2[i-1] + xi[i,j])
                model.addConstr(qc[i,j][0] >= 0)
                model.addConstr(qc[i,j][2] * qc[i,j][2] +  qc[i,j][1] * qc[i,j][1] <= qc[i,j][0] * qc[i,j][0])



    M = 10
    for i in range(N):
        for j in range(N):
            model.addConstr(y[i,j] <= rho1[i])
            model.addConstr(y[i,j] <= M * x[i,j])
            model.addConstr(y[i,j] >= rho1[i] - M * (1-x[i,j]))

            model.addConstr(w[i,j] <= rho2[i])
            model.addConstr(w[i,j] <= M * x[i,j])
            model.addConstr(w[i,j] >= rho2[i] - M * (1-x[i,j]))    
    
    for i in range(N):
        model.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        model.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)

    for i in range(N):
        for j in range(N):
            x[i,j].Start = x_given[i,j]

    model.setParam('OutputFlag', 0)
    # model.setParam('MIPGap',0.01)
    model.setParam('TimeLimit',3600)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro_e.LP")
   
    model.optimize()    
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

    # y_val = np.zeros((N,N))
    # w_val = np.zeros((N,N))
    # for i in range(N):
    #     for j in range(N):
    #         y_val[i,j] = y[i,j].x
    #         w_val[i,j] = w[i,j].x
    # print('y_val',y_val)
    # print('w_val',w_val)

    # rho1_val = np.zeros(N)
    # rho2_val = np.zeros(N)
    # for i in range(N):
    #     rho1_val[i] = rho1[i].x
    #     rho2_val[i] = rho2[i].x
    # print('rho1:',rho1_val)
    # print('rho2:',rho2_val)
    return obj_val, x_seq


def det_release_time_scheduling_given_seq(N,mu,sigma,r,x_seq):
    import heuristic
    x,x_dict = heuristic.decode(x_seq,N)
    mu = x @ mu
    sigma = x @ sigma
    r = x @ r
    s = np.zeros(N)
    for i in range(1,N):
        s[i] = r[i] - r[i-1]

    s[0] = r[0]
    model = gp.Model('det')

    xi = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'xi')
    lbd = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    rho1 = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'rho1')
    rho2 = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'rho2')


    obj = 0
    obj = obj + lbd.sum() 
    for j in range(N):
        obj = obj + rho1[j] * mu[j] + rho2[j] * sigma[j]

    model.setObjective(obj,GRB.MINIMIZE)

    lhs = {}
    for k in range(N):
        for j in range(k,N+1,1):
            max_index = min(j,N-1)
            lhs[k,j] = 0
            for i in range(k,max_index+1):
                # print('k:',k,',j:',j,',i:',i)
                if i == 0:
                    lhs[k,j] = lhs[k,j] + xi[i,j] - lbd[i] - s[i] * (j-i)
                else:
                    lhs[k,j] = lhs[k,j] + xi[i,j] - lbd[i] - s[i] * (j-i)
            model.addConstr(lhs[k,j] <= 0)

    qc = {}
    for i in range(N):
        for j in range(i,N+1):
            name = 'qc['+str(i) + ',' + str(j) + ']'
            qc[i,j] = model.addVars(3,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = name)
            if i == 0:
                model.addConstr(qc[i,j][2] == -rho1[N-1])
                model.addConstr(qc[i,j][1] == rho2[N-1] - xi[i,j])
                model.addConstr(qc[i,j][0] == rho2[N-1] + xi[i,j])
                model.addConstr(qc[i,j][0] >= 0)
                model.addConstr(qc[i,j][2] * qc[i,j][2] +  qc[i,j][1] * qc[i,j][1] <= qc[i,j][0] * qc[i,j][0])

                model.addConstr(xi[i,j] >= 0)
            else:
                model.addConstr(qc[i,j][2] == (j-i) - rho1[i-1])
                model.addConstr(qc[i,j][1] == rho2[i-1] - xi[i,j])
                model.addConstr(qc[i,j][0] == rho2[i-1] + xi[i,j])
                model.addConstr(qc[i,j][0] >= 0)
                model.addConstr(qc[i,j][2] * qc[i,j][2] +  qc[i,j][1] * qc[i,j][1] <= qc[i,j][0] * qc[i,j][0])

    model.setParam('OutputFlag', 0)
    # model.setParam('MIPGap',0.05)
    model.setParam('TimeLimit',600)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    model.optimize()    
    if model.status == 2 or model.status == 13:
        obj_val = model.getObjective().getValue()

    else:
        obj_val = -1000

    # rho1_val = np.zeros(N)
    # rho2_val = np.zeros(N)
    # for i in range(N):
    #     rho1_val[i] = rho1[i].x
    #     rho2_val[i] = rho2[i].x
    # print('rho1:',rho1_val)
    # print('rho2:',rho2_val)
    return obj_val


def det_release_time_scheduling_wass(N,r,c,M,p_hat,d_bar,d_low):


    model = gp.Model('wass')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')
    lbd = model.addVars(M,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
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
    model.setParam('TimeLimit',600)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    model.optimize()    
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

    print('kappa',ka.x)
    return obj_val,x_seq


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
    model.setParam('TimeLimit',600)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    model.optimize()    
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

    print('kappa',ka.x)
    return obj_val,x_seq



def det_release_time_scheduling_RS(N,r,tau,M,p_hat,d_bar,d_low):


    model = gp.Model('det')
    ka = model.addVar(vtype = GRB.CONTINUOUS,lb = 0,name = 'ka')
    theta = model.addVars(M,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'theta')
    lbd = model.addVars(M,N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    y = model.addVars(N,N,vtype = GRB.CONTINUOUS,lb = 0,name = 'y')

    x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')

    model.setObjective(ka,GRB.MINIMIZE)

    model.addConstr(tau * M >= quicksum(theta))

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
                        - quicksum([y[N-1,q]*(d_low[q]-p_hat[q,m]) for q in range(N)])  ) 
                    model.addConstr(v[m][i,j] >= -quicksum([x[i,q] * r[q] for q in range(N)]) * (j-i) \
                        + quicksum([y[N-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)])  ) 
                else:
                    model.addConstr(v[m][i,j] >= -quicksum([(x[i,q] - x[i-1,q]) * r[q] * (j-i) for q in range(N)]) \
                        - quicksum([y[i-1,q]*(d_low[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([(j-i)*x[i-1,q]*d_low[q] for q in range(N)]) ) 
                    model.addConstr(v[m][i,j] >= -quicksum([(x[i,q] - x[i-1,q]) * r[q] * (j-i) for q in range(N)]) \
                        - quicksum([y[i-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([(j-i)*x[i-1,q]*d_bar[q] for q in range(N)]) )
                    model.addConstr(v[m][i,j] >= -quicksum([(x[i,q] - x[i-1,q]) * r[q] * (j-i) for q in range(N)]) \
                        + quicksum([y[i-1,q]*(d_bar[q]-p_hat[q,m]) for q in range(N)]) \
                        + quicksum([(j-i)*x[i-1,q]*d_bar[q] for q in range(N)]) ) 

    M_val = 100000
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
    model.setParam('TimeLimit',600)
    # model.setParam('ConcurrentMIP',6)

    # model.write("E:\\onedrive\\dro.LP")
   
    model.optimize()    
    if model.status == 2 or model.status == 13:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_tem[i,j] = x[i,j].x
        x_seq = x_tem @ np.arange(N)

    else:
        obj_val = -1000

 
    return obj_val,x_seq



def rand_releas_time_scheduling_given_X(N,mu_p,sigma_p,mu_r,sigma_r):
    mu_p = np.round(mu_p,1)
    sigma_p = np.round(sigma_p,1)
    mu_r = np.round(mu_r,1)
    sigma_r = np.round(sigma_r,1)

    model = gp.Model('det')
    xi = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'xi')
    phi = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'phi')
    lbd = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    rho1_p = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'rho1_p')
    rho2_p = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'rho2_p')
    rho1_s = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'rho1_s')
    rho2_s = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'rho2_s')


    obj = 0
    obj = obj + lbd.sum()
    for j in range(N):
        obj = obj + rho1_p[j] * mu_p[j] + rho2_p[j] * sigma_p[j] \
            + rho1_s[j] * mu_r[j] + rho2_s[j] * sigma_r[j]

    model.setObjective(obj,GRB.MINIMIZE)


    lhs = {}
    for k in range(N):
        for j in range(k,N+1,1):
            max_index = min(j,N-1)
            lhs[k,j] = 0
            for i in range(k,max_index+1):
                # print('k:',k,',j:',j,',i:',i)
                lhs[k,j] = lhs[k,j] + xi[i,j] + phi[i,j] - lbd[i] 
            model.addConstr(lhs[k,j] <= 0)

    qc_p = {}
    qc_s = {}
    for i in range(N):
        for j in range(i,N+1):
            name = 'qc_p['+str(i) + ',' + str(j) + ']'
            qc_p[i,j] = model.addVars(3,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = name)
            model.addConstr(qc_p[i,j][2] == (j-i) - rho1_p[i])
            model.addConstr(qc_p[i,j][1] == rho2_p[i] - xi[i,j])
            model.addConstr(qc_p[i,j][0] == rho2_p[i] + xi[i,j])
            model.addConstr(qc_p[i,j][0] >= 0)
            model.addConstr(qc_p[i,j][2] * qc_p[i,j][2] +  qc_p[i,j][1] * qc_p[i,j][1] <= qc_p[i,j][0] * qc_p[i,j][0])

            name = 'qc_s['+str(i) + ',' + str(j) + ']'
            qc_s[i,j] = model.addVars(3,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = name)
            model.addConstr(qc_s[i,j][2] == -(j-i) - rho1_s[i])
            model.addConstr(qc_s[i,j][1] == rho2_s[i] - phi[i,j])
            model.addConstr(qc_s[i,j][0] == rho2_s[i] + phi[i,j])
            model.addConstr(qc_s[i,j][0] >= 0)
            model.addConstr(qc_s[i,j][2] * qc_s[i,j][2] +  qc_s[i,j][1] * qc_s[i,j][1] <= qc_s[i,j][0] * qc_s[i,j][0])




    model.setParam('OutputFlag', 1)
    # model.setParam('TimeLimit',600)
    model.write("E:\\onedrive\\dro.LP")
   
    model.optimize()    
    if model.status == 2 or model.status == 13:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        # x_seq = x_tem @ np.arange(N)
    else:
        obj_val = -1000
        # x_seq = np.zeros(N)   
    
    return obj_val


def rong_scheduling(N,mu,sigma,r):
    mu = np.round(mu,1)
    sigma = np.round(sigma,1)
    r = np.round(r,1)

    # ***** gurobi version 
    model = gp.Model('dro')
    # x = model.addVars(N,N,vtype = GRB.BINARY,name = 'x')
    xi = model.addVars(N,N+1,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'xi')
    lbd = model.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = 'lbd')
    rho1 = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'rho1')
    rho2 = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 'rho2')
    s = model.addVars(N,vtype = GRB.CONTINUOUS,lb = 0,name = 's')


    obj = 0
    obj = obj + lbd.sum()
    for i in range(N):
        obj = obj + rho1[i] * mu[i] + rho2[i] * sigma[i]
    
    model.setObjective(obj,GRB.MINIMIZE)

    lhs = {}
    for k in range(N):
        for j in range(k,N+1,1):
            max_index = min(j,N-1)
            lhs[k,j] = 0
            for i in range(k,max_index+1):
                print('k:',k,',j:',j,',i:',i)
                lhs[k,j] = lhs[k,j] + xi[i,j] - lbd[i] - s[i] * (j-i) 
            model.addConstr(lhs[k,j] <= 0)

    qc = {}
    for i in range(N):
        for j in range(i,N+1):
            name = 'qc['+str(i) + ',' + str(j) + ']'
            qc[i,j] = model.addVars(3,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name = name)
            model.addConstr(qc[i,j][2] == (j-i) - rho1[i])
            model.addConstr(qc[i,j][1] == rho2[i] - xi[i,j])
            model.addConstr(qc[i,j][0] == rho2[i] + xi[i,j])
            model.addConstr(qc[i,j][0] >= 0)
            model.addConstr(qc[i,j][2] * qc[i,j][2] +  qc[i,j][1] * qc[i,j][1] <= qc[i,j][0] * qc[i,j][0])
    
    model.addConstr(s.sum() <= mu.sum())
    

    model.setParam('OutputFlag', 1)
    # model.setParam('TimeLimit',600)
    model.write("E:\\onedrive\\dro.LP")
   
    model.optimize()   
    
     
    if model.status == 2 or model.status == 13:
        obj_val = model.getObjective().getValue()
        x_seq = np.zeros((N,N))
        # for i in range(N):
        #     for j in range(N):
        #         x_tem[i,j] = x[i,j].x
        # x_seq = x_tem @ np.arange(N)
    else:
        obj_val = -1000
        x_seq = np.zeros(N)   
    

    # # rsome version
    # model = ro.Model()
    # xi = model.dvar((N,N+1))
    # lbd = model.dvar(N)
    # rho1 = model.dvar(N)
    # rho2 = model.dvar(N)
    # s = model.dvar(N)



    # obj = 0
    # obj = obj + lbd.sum()
    # for i in range(N):
    #     obj = obj + rho1[i] * mu[i] + rho2[i] * sigma[i]
    
    # model.min(obj)

    # model.st(rho2 >= 0)
    # model.st(s >= 0)

    # lhs = {}
    # for k in range(N):
    #     for j in range(k,N+1,1):
    #         max_index = min(j,N-1)
    #         lhs[k,j] = 0
    #         for i in range(k,max_index+1):
    #             print('k:',k,',j:',j,',i:',i)
    #             lhs[k,j] = lhs[k,j] + xi[i,j] - lbd[i] - s[i] * (j-i) 
    #         model.st(lhs[k,j] <= 0)

    # qc = {}
    # for i in range(N):
    #     for j in range(i,N+1):
    #         qc[i,j] = model.dvar(2)
    #         model.st((j-i) - rho1[i] == qc[i,j][1])
    #         model.st(rho2[i] - xi[i,j] == qc[i,j][0])
    #         model.st(norm(qc[i,j]) <= rho2[i] + xi[i,j])

    #         # model.addConstr(rho2[i] + xi[i,j] >= 0)
    
    # model.st(s.sum() <= mu.sum())
    # model.solve(grb,display = True)
    # s_val = s.get()

    # model.setParam('OutputFlag', 1)
    # # model.setParam('TimeLimit',600)
    # model.write("E:\\onedrive\\dro.LP")
   
    # model.optimize()   


    return 1


