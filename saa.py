# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 08:49:12 2020

@author: xunzhang
"""
##


import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from gurobipy import quicksum


#p = input_p;
#r_hat = r_hat_input;



#p_dict = sio.loadmat('p.mat')
#p_tem = p_dict['p']
#p = np.zeros(n);
#for i in range(n):
#    p[i] = np.asscalar(p_tem[i])
#
#r_hat_dict = sio.loadmat('r_hat.mat')
#r_hat_tem = r_hat_dict['r_hat']
#r_hat = np.zeros((n,k));
#for i in range(n):
#    for j in range(k):
#        r_hat[i,j] = np.asscalar(r_hat_tem[i,j])

def saa_seq(N,k,p_hat,r_hat):
    e = (1/k)*np.ones(k)
    m = gp.Model("saa")
    
    t = m.addVars(N,k,vtype = gp.GRB.CONTINUOUS, lb = -GRB.INFINITY, name='t')
    
    x = m.addVars(N, N, vtype = gp.GRB.BINARY,name='x')
    
    xp = m.addVars(N, k, vtype = gp.GRB.CONTINUOUS,lb = -GRB.INFINITY,name='xp')
    te = m.addVars(N, vtype = gp.GRB.CONTINUOUS,lb = -GRB.INFINITY,name='te')
    xpe = m.addVars(N, vtype = gp.GRB.CONTINUOUS,lb = -GRB.INFINITY,name='xpe')
    
    m.setObjective(te.sum() + xpe.sum(), GRB.MINIMIZE )
    
    for i in range(N):
        for j in range(k):
            m.addConstr(t[i,j] >= quicksum([x[i,ind] * r_hat[ind,j] for ind in range(N)]) )
            m.addConstr(xp[i,j] == quicksum([x[i,ind] * p_hat[ind,j] for ind in range(N)]) )
            
    for j in range(k):
        for i in range(1,N):
            m.addConstr(t[i,j] >= t[i-1,j] + quicksum([x[i-1,ind] * p_hat[ind,j] for ind in range(N)]) )
    
    
    for i in range(N):
        m.addConstr(te[i] == quicksum([t[i,ind] * e[ind] for ind in range(k)]))        
        m.addConstr(xpe[i] == quicksum([xp[i,ind] * e[ind] for ind in range(k)]))   
        # At most one queen per row
        m.addConstr(quicksum([x[i,ind] for ind in range(N)]) == 1)
        # At most one queen per column
        m.addConstr(quicksum([x[ind,i] for ind in range(N)]) == 1 )

    # for i in range(N):
    #     for j in range(N):
    #         x[i,j].Start = x_given[i,j]
    m.setParam('OutputFlag', 0)
    # m.setParam('MIPGap',0.01)
    # m.setParam('TimeLimit',600)
    start = time.time()
    m.optimize()
    end = time.time()
    saa_cpu_time = end - start
    # m.write("saa.LP")
    x_result = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            x_result[i,j] = x[i,j].x
    x_result = np.asmatrix(x_result)
    joblist = range(1,N+1)
    schedule_tem = np.dot(x_result,joblist)
    schedule = np.zeros(N)
    for i in range(N):
        schedule[i] = schedule_tem[0,i]
    obj = m.getObjective().getValue()
    return schedule,obj,saa_cpu_time



def saa_seq_det_release(N,k,p_hat,r):
    e = (1/k)*np.ones(k)
    m = gp.Model("saa")
    
    t = m.addVars(N,k,vtype = gp.GRB.CONTINUOUS, lb = -GRB.INFINITY, name='t')
    x = m.addVars(N, N, vtype = gp.GRB.BINARY,name='x')
    
    xp = m.addVars(N, k, vtype = gp.GRB.CONTINUOUS,lb = -GRB.INFINITY,name='xp')
    # te = m.addVars(N, vtype = gp.GRB.CONTINUOUS,lb = -GRB.INFINITY,name='te')
    # xpe = m.addVars(N, vtype = gp.GRB.CONTINUOUS,lb = -GRB.INFINITY,name='xpe')
    
    obj = 0
    for j in range(k):
        obj = obj + (1/k)*(quicksum([t[i,j] + xp[i,j] for i in range(N)]))
    m.setObjective(obj, GRB.MINIMIZE )
    

    for i in range(N):
        for j in range(k):
            m.addConstr(t[i,j] >= quicksum([x[i,ind] * r[ind] for ind in range(N)]) )
            m.addConstr(xp[i,j] == quicksum([x[i,ind] * p_hat[ind,j] for ind in range(N)]) )
            
    for j in range(k):
        for i in range(1,N):
            m.addConstr(t[i,j] >= t[i-1,j] + quicksum([x[i-1,ind] * p_hat[ind,j] for ind in range(N)]) )
    
    
    for i in range(N):
    #     m.addConstr(te[i] == quicksum([t[i,ind] * e[ind] for ind in range(k)]))        
    #     m.addConstr(xpe[i] == quicksum([xp[i,ind] * e[ind] for ind in range(k)]))   
        # At most one queen per row
        m.addConstr(quicksum([x[i,ind] for ind in range(N)]) == 1)
        # At most one queen per column
        m.addConstr(quicksum([x[ind,i] for ind in range(N)]) == 1 )

 
    m.setParam('OutputFlag', 1)
    m.setParam('MIPGap',0.01)
    m.setParam('TimeLimit',600)
    start = time.time()
    m.optimize()
    end = time.time()
    saa_cpu_time = end - start
    if m.status == 2 or m.status == 13 or m.status == 9:

        # m.write("saa.LP")
        x_result = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                x_result[i,j] = x[i,j].x
        x_result = np.asmatrix(x_result)
        joblist = range(1,N+1)
        schedule_tem = np.dot(x_result,joblist)
        schedule = np.zeros(N)
        for i in range(N):
            schedule[i] = schedule_tem[0,i]
        obj = m.getObjective().getValue()
    
    return schedule,obj,saa_cpu_time


