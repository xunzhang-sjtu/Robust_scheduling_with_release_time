
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:31:55 2020

@author: xunzhang
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

#N = 6
#mu = np.asarray([42.5889,46.2317,15.0795,46.5350,35.2944,13.9016,172.2901,87.3676,144.0505,25.5395,75.9170,164.8324])
#sigma = np.asarray([11.8609,25.2833,14.4387,44.9011,5.5629,13.4928,136.4894,83.8286,94.4598,0.9121,64.4634,153.9523])


#N = 3
#mu = np.asarray([37.7931,22.6840,48.0089,68.8965,71.5680,16.8185])
#sigma = np.asarray([1.3018,9.9525,18.3182,33.7431,31.8897,10.8700])
def lldr_mad_seq(N,mu,mad):

    #    xe = { (x) : 1 for x in range(N)}
    #    r_mu_dict ={ (x) : r_mu[x] for x in range(N)}
    
    m = gp.Model("dro_mad")
    
    s = m.addMVar(2*N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='s')
    w = m.addMVar(2*N, vtype = GRB.CONTINUOUS,lb = 0,ub = GRB.INFINITY,name='o')
    v = m.addMVar(1, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='v')
    
    t0 = m.addMVar(N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t0')
    t1 = m.addMVar((N,2*N), vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t1')
    t2 = m.addMVar((N,2*N), vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t2')
    
    alp = m.addMVar(2*N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name='alp')
    bet = m.addMVar(2*N, vtype = GRB.CONTINUOUS,lb = 0,name='bet')
    
    a = m.addMVar((2*N, N), vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name='a')
    b = m.addMVar((2*N, N), vtype = GRB.CONTINUOUS,lb = 0,name='b')
    c = m.addMVar((2*N, N-1), vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,ub = 0, name='c')    
    d = m.addMVar((2*N, N-1), vtype = GRB.CONTINUOUS,lb = 0,name='d')

    x = m.addMVar((N, N), vtype = GRB.BINARY,name='x')   
#    x = m.addMVar((N, N), vtype = GRB.CONTINUOUS,name='x') # relax binary constraint
    
    m.setObjective(s @ mu + w @ mad + v, GRB.MINIMIZE )
    
    
    m.addConstr(v - t0.sum() >= alp @ mu + bet @ mu)
    
    for j in range(N):
        m.addConstr(alp[j] + bet[j] >= t1[:,j].sum()+ x[:,j].sum()-s[j])
    
    for j in range(N,2*N):
        m.addConstr(alp[j] + bet[j] >= t1[:,j].sum()-s[j])
    
    for j in range(2*N):
        m.addConstr(alp[j] - bet[j] == t2[:,j].sum() - w[j])
        
    for i in range(N):
        m.addConstr(t0[i] >= a[:,i] @ mu + b[:,i] @ mu) 
    
    for i in range(N):
        for j in range(N):
            m.addConstr(a[j,i] + b[j,i] >= -t1[i,j])
    
    for i in range(N):
        for j in range(N,2*N):
            m.addConstr(a[j,i] + b[j,i] >= x[i,j-N]-t1[i,j])
    
    for i in range(N):
        for j in range(2*N):
            m.addConstr(a[j,i] - b[j,i] == -t2[i,j])
    
    for i in range(1,N):
        m.addConstr(t0[i]-t0[i-1] >= c[:,i-1] @ mu + d[:,i-1] @ mu)
    
    for i in range(1,N):
        for j in range(N):
            m.addConstr(c[j,i-1] + d[j,i-1] >= t1[i-1,j] - t1[i,j] - x[i-1,j])
    
    for i in range(1,N):
        for j in range(N,2*N):
            m.addConstr(c[j,i-1] + d[j,i-1] >= t1[i-1,j] - t1[i,j])
            
    for i in range(1,N):
        for j in range(2*N):
            m.addConstr(c[j,i-1] - d[j,i-1] >= t2[i-1,j] - t2[i,j])         
                
    
    for i in range(N):
        m.addConstr(x[i,:].sum() == 1)
        m.addConstr(x[:,i].sum() == 1)
        
    m.setParam('OutputFlag', 0)
    m.optimize()
    #m.write("IB.LP")
    
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
    return schedule,obj

