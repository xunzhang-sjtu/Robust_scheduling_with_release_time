# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:31:55 2020

@author: xunzhang
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
N = 6
mu = np.random.uniform(10,20,2*N)
sigma = np.random.uniform(0,15*N,2*N)
#mu = np.asarray([42.5889,46.2317,15.0795,46.5350,35.2944,13.9016,172.2901,87.3676,144.0505,25.5395,75.9170,164.8324])
#sigma = np.asarray([11.8609,25.2833,14.4387,44.9011,5.5629,13.4928,136.4894,83.8286,94.4598,0.9121,64.4634,153.9523])



def lldr_seq(N,mu,sigma):
    N = 6
#    mu = np.asarray([20,10,30])
#    sigma = np.asarray([10,1,20])
    mu = np.random.uniform(10,20,2*N)
    sigma = np.random.uniform(0,15*N,2*N)
    sigma_square = sigma * sigma
    #    xe = { (x) : 1 for x in range(N)}
    #    r_mu_dict ={ (x) : r_mu[x] for x in range(N)}
    
    m = gp.Model("liftedLdr")
    
    s = m.addMVar(N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='s')
    o = m.addMVar(N, vtype = GRB.CONTINUOUS,lb = 0,ub = GRB.INFINITY,name='o')
    v = m.addMVar(1, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='v')
    
    t0 = m.addMVar(N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t0')
    t1 = m.addMVar((N,N), vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t1')
    t2 = m.addMVar((N,N), vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t2')
    
    alp = m.addMVar(N, vtype = GRB.CONTINUOUS,name='alp')
    bet = m.addMVar(N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='bet')
    tao = m.addMVar(N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='tao')
    
    
    d = m.addMVar((N, N), vtype = GRB.CONTINUOUS,name='d')
    e = m.addMVar((N, N), vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='e')
    f = m.addMVar((N, N), vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='f')
    
    x = m.addMVar((N, N), vtype = GRB.BINARY,name='x')   
#    x = m.addMVar((N, N), vtype = GRB.CONTINUOUS,name='x') # relax binary constraint
    
    m.setObjective(s @ mu + o @ sigma_square + v, GRB.MINIMIZE )
    
    
    m.addConstr(v - t0.sum() >= 0.5*alp.sum() - 0.5*bet.sum() - tao @ mu)
    
    for j in range(N):
        m.addConstr(tao[j] <= s[j] - t1[:,j].sum() - 1)
    
    for j in range(N):
        m.addConstr(0.5*alp[j] + 0.5*bet[j] == o[j] - t2[:,j].sum())
        m.addConstr(bet[j]@bet[j] + tao[j]@tao[j] <= alp[j]@alp[j])
        
        
    m.addConstr(t0[0] >= 0.5*d[:,0].sum()-0.5*e[:,0].sum() - f[:,0] @ mu) 
    for j in range(N):
        m.addConstr(f[j,0] <= t1[0,j])
        
    for j in range(N):
        m.addConstr(0.5*d[j,0] + 0.5*e[j,0] == t2[0,j]) 
        m.addConstr(e[j,0]@e[j,0] + f[j,0]@f[j,0] <= d[j,0]@d[j,0])
        
        
        
    for i in range(1,N):
        m.addConstr(t0[i]-t0[i-1] >= 0.5*d[:,i].sum() - 0.5*e[:,i].sum() - f[:,i] @ mu)
    
    for i in range(1,N):
        for j in range(N):
            m.addConstr(f[j,i] <= t1[i,j] - t1[i-1,j] - x[i-1,j])

    for i in range(1,N):
        for j in range(N):
            m.addConstr(0.5*d[j,i] + 0.5*e[j,i] == t2[i,j] - t2[i-1,j])         
            m.addConstr(e[j,i]@e[j,i] + f[j,i]@f[j,i] <= d[j,i]@d[j,i])
                
    
    for i in range(N):
        m.addConstr(x[i,:].sum() == 1)
        m.addConstr(x[:,i].sum() == 1)
        
    m.setParam('OutputFlag', 1)
    start = time.time()
    m.optimize()
    end = time.time()
    lldr_cpu_time = end - start
    #m.write("IB.LP")
    
    
    
    d_result = np.zeros((N,N));
    for i in range(N):
        for j in range(N):
            d_result[i,j] = d[i,j].x
    e_result = np.zeros((N,N));
    for i in range(N):
        for j in range(N):
            e_result[i,j] = e[i,j].x            
    f_result = np.zeros((N,N));
    for i in range(N):
        for j in range(N):
            f_result[i,j] = f[i,j].x
    t0_result = np.zeros(N);
    for i in range(N):
        t0_result[i] = t0[i].x
    t1_result = np.zeros((N,N));
    for i in range(N):
        for j in range(N):
            t1_result[i,j] = t1[i,j].x
    t2_result = np.zeros((N,N));
    for i in range(N):
        for j in range(N):
            t2_result[i,j] = t2[i,j].x
    o_result = np.zeros(N)
    alpha_result = np.zeros(N)
    beta_result = np.zeros(N)
    tao_result = np.zeros(N)
    for i in range(N):
        o_result[i] = o[i].x
        alpha_result[i] = alp[i].x
        beta_result[i] = bet[i].x
        tao_result = tao[i].x
    
    x_result = np.zeros((N,N));
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
    return schedule,obj,lldr_cpu_time
#i=2
#t0_result[i]+np.dot(t1_result[i,:],np.ones(2*N)) + np.dot(t2_result[i,:],np.ones(2*N))
    


def lldr_seq_no_release(N,mu,sigma):
    N = 3
    mu = np.asarray([10.1,10.2,10.3])
    sigma = np.asarray([10,1,5])
    cv = sigma/mu
    mu = mu + cv
    sigma_square = sigma * sigma
    #    xe = { (x) : 1 for x in range(N)}
    #    r_mu_dict ={ (x) : r_mu[x] for x in range(N)}
    
    m = gp.Model("liftedLdr")
    
    s = m.addMVar(N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='s')
    o = m.addMVar(N, vtype = GRB.CONTINUOUS,lb = 0,ub = GRB.INFINITY,name='o')
    v = m.addMVar(1, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='v')
    
    t0 = m.addMVar(N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t0')
    t1 = m.addMVar((N,N), vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t1')
    t2 = m.addMVar((N,N), vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t2')
    
    alp = m.addMVar(N, vtype = GRB.CONTINUOUS,name='alp')
    bet = m.addMVar(N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='bet')
    tao = m.addMVar(N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='tao')
    
    
    d = m.addMVar((N, N), vtype = GRB.CONTINUOUS,name='d')
    e = m.addMVar((N, N), vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='e')
    f = m.addMVar((N, N), vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='f')
    
    x = m.addMVar((N, N), vtype = GRB.BINARY,name='x')   
#    x = m.addMVar((N, N), vtype = GRB.CONTINUOUS,name='x') # relax binary constraint
    
    m.setObjective(s @ mu + o @ sigma_square + v, GRB.MINIMIZE )
    
    
    m.addConstr(v - t0.sum() >= 0.5*alp.sum() - 0.5*bet.sum() - tao @ mu)
    
    for j in range(N):
        m.addConstr(tao[j] <= s[j] - t1[:,j].sum() - 1)
    
    for j in range(N):
        m.addConstr(0.5*alp[j] + 0.5*bet[j] == o[j] - t2[:,j].sum())
        m.addConstr(bet[j]@bet[j] + tao[j]@tao[j] <= alp[j]@alp[j])
        
        
    m.addConstr(t0[0] >= 0.5*d[:,0].sum()-0.5*e[:,0].sum() - f[:,0] @ mu) 
    for j in range(N):
        m.addConstr(f[j,0] <= t1[0,j])
        
    for j in range(N):
        m.addConstr(0.5*d[j,0] + 0.5*e[j,0] == t2[0,j]) 
        m.addConstr(e[j,0]@e[j,0] + f[j,0]@f[j,0] <= d[j,0]@d[j,0])
        
        
        
    for i in range(1,N):
        m.addConstr(t0[i]-t0[i-1] >= 0.5*d[:,i].sum() - 0.5*e[:,i].sum() - f[:,i] @ mu)
    
    for i in range(1,N):
        for j in range(N):
            m.addConstr(f[j,i] <= t1[i,j] - t1[i-1,j] - x[i-1,j])

    for i in range(1,N):
        for j in range(N):
            m.addConstr(0.5*d[j,i] + 0.5*e[j,i] == t2[i,j] - t2[i-1,j])         
            m.addConstr(e[j,i]@e[j,i] + f[j,i]@f[j,i] <= d[j,i]@d[j,i])
                
    
    for i in range(N):
        m.addConstr(x[i,:].sum() == 1)
        m.addConstr(x[:,i].sum() == 1)
        
    m.setParam('OutputFlag', 1)
    start = time.time()
    m.optimize()
    end = time.time()
    lldr_cpu_time = end - start
    #m.write("IB.LP")
    
    
    
    d_result = np.zeros((N,N));
    for i in range(N):
        for j in range(N):
            d_result[i,j] = d[i,j].x
    e_result = np.zeros((N,N));
    for i in range(N):
        for j in range(N):
            e_result[i,j] = e[i,j].x            
    f_result = np.zeros((N,N));
    for i in range(N):
        for j in range(N):
            f_result[i,j] = f[i,j].x
    t0_result = np.zeros(N);
    for i in range(N):
        t0_result[i] = t0[i].x
    t1_result = np.zeros((N,N));
    for i in range(N):
        for j in range(N):
            t1_result[i,j] = t1[i,j].x
    t2_result = np.zeros((N,N));
    for i in range(N):
        for j in range(N):
            t2_result[i,j] = t2[i,j].x
    o_result = np.zeros(N)
    alpha_result = np.zeros(N)
    beta_result = np.zeros(N)
    tao_result = np.zeros(N)
    for i in range(N):
        o_result[i] = o[i].x
        alpha_result[i] = alp[i].x
        beta_result[i] = bet[i].x
        tao_result = tao[i].x
    
    x_result = np.zeros((N,N));
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
    return schedule,obj,lldr_cpu_time
#i=2
#t0_result[i]+np.dot(t1_result[i,:],np.ones(2*N)) + np.dot(t2_result[i,:],np.ones(2*N))