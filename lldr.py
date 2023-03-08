# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 18:31:55 2020

@author: xunzhang
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
from gurobipy import quicksum

# N = 6
# mu = np.random.uniform(10,20,2*N)
# sigma = np.random.uniform(0,15*N,2*N)
#mu = np.asarray([42.5889,46.2317,15.0795,46.5350,35.2944,13.9016,172.2901,87.3676,144.0505,25.5395,75.9170,164.8324])
#sigma = np.asarray([11.8609,25.2833,14.4387,44.9011,5.5629,13.4928,136.4894,83.8286,94.4598,0.9121,64.4634,153.9523])



def lldr_seq(N,mu,sigma,x_given):
#    N = 3
#    mu = np.asarray([20,10,30,5,15,25])
#    sigma = np.asarray([0.1,0.1,0.1,10,10,10])
    
    sigma_square = sigma * sigma
    #    xe = { (x) : 1 for x in range(N)}
    #    r_mu_dict ={ (x) : r_mu[x] for x in range(N)}
    
    m = gp.Model("liftedLdr")
    
    s = m.addVars(2*N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='s')
    o = m.addVars(2*N, vtype = GRB.CONTINUOUS,lb = 0,ub = GRB.INFINITY,name='o')
    v = m.addVar(vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='v')
    
    t0 = m.addVars(N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t0')
    t1 = m.addVars(N,2*N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t1')
    t2 = m.addVars(N,2*N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t2')
    
    alp = m.addVars(2*N, vtype = GRB.CONTINUOUS,name='alp')
    bet = m.addVars(2*N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='bet')
    tao = m.addVars(2*N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='tao')
    
    a = m.addVars(2*N, N, vtype = GRB.CONTINUOUS,name='a')
    b = m.addVars(2*N, N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='b')
    c = m.addVars(2*N, N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='c')
    
    d = m.addVars(2*N, N-1, vtype = GRB.CONTINUOUS,name='d')
    e = m.addVars(2*N, N-1, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='e')
    f = m.addVars(2*N, N-1, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='f')
    
    x = m.addVars(N, N, vtype = GRB.BINARY,name='x')   
    # x = m.addVars(N, N, vtype = GRB.CONTINUOUS,lb = 0,name='x') # relax binary constraint
    
    m.setObjective(quicksum([s[i]* mu[i] + o[i] * sigma_square[i] for i in range(2*N)]) + v, GRB.MINIMIZE )
    
    
    m.addConstr(v - t0.sum() >= 0.5*alp.sum() - 0.5*bet.sum() - quicksum([tao[i] * mu[i] for i in range(2*N)]))
    
    for j in range(N):
        m.addConstr(tao[j] <= s[j] - quicksum([t1[i,j] for i in range(N)]) - 1)
    
    for j in range(N,2*N):
        m.addConstr(tao[j] <= s[j] - quicksum([t1[i,j] for i in range(N)]))
    
    for j in range(2*N):
        m.addConstr(0.5*alp[j] + 0.5*bet[j] == o[j] - quicksum([t1[i,j] for i in range(N)]))
        
    for i in range(N):
        m.addConstr(t0[i] >= 0.5*quicksum([a[j,i]-b[j,i] -2*c[j,i] * mu[j] for j in range(2*N)]) ) 
    
    for i in range(N):
        for j in range(N):
            m.addConstr(c[j,i] <= t1[i,j])
    
    for i in range(N):        
        for j in range(N,2*N):
            m.addConstr(c[j,i] <= t1[i,j] - x[i,j-N])
    
    
    for i in range(N):
        for j in range(2*N):
            m.addConstr(0.5*a[j,i] + 0.5*b[j,i] == t2[i,j])
    
    for i in range(N):
        for j in range(2*N):
            m.addConstr(b[j,i]*b[j,i] + c[j,i]*c[j,i] <= a[j,i]*a[j,i])
    
    for j in range(2*N):
        m.addConstr(bet[j]*bet[j] + tao[j]*tao[j] <= alp[j]*alp[j])
    
    for i in range(1,N):
        m.addConstr(t0[i]-t0[i-1] >= 0.5*quicksum([d[j,i-1]-e[j,i-1] - 2 *f[j,i-1] * mu[j] for j in range(2*N)]))
    
    for i in range(1,N):
        for j in range(N):
            m.addConstr(f[j,i-1] <= t1[i,j] - t1[i-1,j] - x[i-1,j])
    
    for i in range(1,N):
        for j in range(N,2*N):
            m.addConstr(f[j,i-1] <= t1[i,j] - t1[i-1,j])
            
    for i in range(1,N):
        for j in range(2*N):
            m.addConstr(0.5*d[j,i-1] + 0.5*e[j,i-1] == t2[i,j] - t2[i-1,j])         
            m.addConstr(e[j,i-1]*e[j,i-1] + f[j,i-1]*f[j,i-1] <= d[j,i-1]*d[j,i-1])
                
    
    for i in range(N):
        m.addConstr(quicksum(x[i,j] for j in range(N)) == 1)
        m.addConstr(quicksum(x[j,i] for j in range(N)) == 1)
    
    for i in range(N):
        for j in range(N):
            x[i,j].Start = x_given[i,j]

    m.setParam('OutputFlag', 1)
    m.setParam('MIPGap',0.01)
    m.setParam('TimeLimit',600)
    start = time.time()
    m.optimize()
    end = time.time()
    lldr_cpu_time = end - start
    #m.write("IB.LP")
    # dv = m.getConstrByName("convexity").getAttr("Pi")
    
    
    
    a_result = np.zeros((2*N,N));
    for i in range(2*N):
        for j in range(N):
            a_result[i,j] = a[i,j].x
    b_result = np.zeros((2*N,N));
    for i in range(2*N):
        for j in range(N):
            b_result[i,j] = b[i,j].x
    c_result = np.zeros((2*N,N));
    for i in range(2*N):
        for j in range(N):
            c_result[i,j] = c[i,j].x
    d_result = np.zeros((2*N,N-1));
    for i in range(2*N):
        for j in range(N-1):
            d_result[i,j] = d[i,j].x
    e_result = np.zeros((2*N,N-1));
    for i in range(2*N):
        for j in range(N-1):
            e_result[i,j] = e[i,j].x            
    f_result = np.zeros((2*N,N-1));
    for i in range(2*N):
        for j in range(N-1):
            f_result[i,j] = f[i,j].x
    t0_result = np.zeros(N);
    for i in range(N):
        t0_result[i] = t0[i].x
    t1_result = np.zeros((N,2*N));
    for i in range(N):
        for j in range(2*N):
            t1_result[i,j] = t1[i,j].x
    t2_result = np.zeros((N,2*N));
    for i in range(N):
        for j in range(2*N):
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


def DRO_seq(N,mu,sigma,x_given):
#    N = 3
#    mu = np.asarray([20,10,30,5,15,25])
#    sigma = np.asarray([0.1,0.1,0.1,1,1,1])
    
    sigma_square = sigma * sigma

    m = gp.Model("liftedLdr")
    
    
    t0 = m.addVars(N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t0')
    t1 = m.addVars(N,2*N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t1')
    t2 = m.addVars(N,2*N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t2')
    

    a = m.addVars(2*N, N, vtype = GRB.CONTINUOUS,lb = 0,name='a')
    c = m.addVars(2*N, N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='c')
    
    d = m.addVars(2*N, N, vtype = GRB.CONTINUOUS,lb = 0,name='d')
    f = m.addVars(2*N, N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='f')
    
    x = m.addVars(N, N, vtype = GRB.BINARY,name='x')   
    # x = m.addVars(N, N, vtype = GRB.CONTINUOUS,lb = 0,name='x') # relax binary constraint
    
    obj = 0
    for i in range(N):
        obj = obj + quicksum([t1[i,j] * mu[j] + t2[i,j]*sigma_square[j] for j in range(2*N)])
    obj = obj + quicksum(t0)

    m.setObjective(obj, GRB.MINIMIZE)
    

    qc1 = {}
    qc2 = {}
    for i in range(N):
        m.addConstr(t0[i] >= quicksum([a[j,i] - c[j,i] * mu[j] for j in range(2*N)]) )

        for j in range(2*N):
            if j < N:
                m.addConstr(c[j,i] == t1[i,j])
            else:
                m.addConstr(c[j,i] == t1[i,j] - x[i,j-N])
            name = 'qc1_' + str(i) + str(j)
            qc1[i,j] = m.addVars(3, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name=name)
            m.addConstr(qc1[i,j][0] == a[j,i] + t2[i,j] )
            m.addConstr(qc1[i,j][1] == a[j,i] - t2[i,j] )
            m.addConstr(qc1[i,j][2] == c[j,i])
            m.addConstr(qc1[i,j][2]*qc1[i,j][2] + qc1[i,j][1] * qc1[i,j][1] <= qc1[i,j][0]*qc1[i,j][0] )
            m.addConstr(qc1[i,j][0] >= 0) 


        if i == 0:
            m.addConstr(t0[i] >= quicksum([d[j,i] - f[j,i] * mu[j] for j in range(2*N)]) )

            for j in range(2*N):
                name = 'qc2_' + str(i) + str(j)
                qc2[i,j] = m.addVars(3, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name=name)
                if j < N:
                    m.addConstr(f[j,i] == t1[i,j])
                else:
                    m.addConstr(f[j,i] == t1[i,j])
                # m.addConstr(4 * t2[i,j] * d[j,i] >= f[j,i] * f[j,i])

                m.addConstr(qc2[i,j][0] == d[j,i] + t2[i,j] )
                m.addConstr(qc2[i,j][1] == d[j,i] - t2[i,j] )
                m.addConstr(qc2[i,j][2] == f[j,i])
                m.addConstr(qc2[i,j][2]*qc2[i,j][2] + qc2[i,j][1] * qc2[i,j][1] <= qc2[i,j][0]*qc2[i,j][0] )
                m.addConstr(qc2[i,j][0] >= 0) 
        else:
            m.addConstr(t0[i] - t0[i-1] >= quicksum([d[j,i] - f[j,i] * mu[j]]) )

            for j in range(2*N):
                name = 'qc2_' + str(i) + str(j)
                qc2[i,j] = m.addVars(3, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name=name)
                if j < N:
                    m.addConstr(f[j,i] == t1[i,j] - t1[i-1,j] - x[i-1,j])
                else:
                    m.addConstr(f[j,i] == t1[i,j] - t1[i-1,j])

                m.addConstr(qc2[i,j][0] == d[j,i] + t2[i,j] - t2[i-1,j] )
                m.addConstr(qc2[i,j][1] == d[j,i] - t2[i,j] + t2[i-1,j] )
                m.addConstr(qc2[i,j][2] == f[j,i])
                m.addConstr(qc2[i,j][2]*qc2[i,j][2] + qc2[i,j][1] * qc2[i,j][1] <= qc2[i,j][0]*qc2[i,j][0] )
                m.addConstr(qc2[i,j][0] >= 0) 




    for i in range(N):
        m.addConstr(quicksum(x[i,j] for j in range(N)) == 1)
        m.addConstr(quicksum(x[j,i] for j in range(N)) == 1)
    
    # for i in range(N):
    #     for j in range(N):
    #         x[i,j].Start = x_given[i,j]

    m.setParam('OutputFlag', 1)
    m.setParam('MIPGap',0.01)
    m.setParam('TimeLimit',600)
    start = time.time()
    m.optimize()
    end = time.time()
    lldr_cpu_time = end - start
    #m.write("IB.LP")
    # dv = m.getConstrByName("convexity").getAttr("Pi")

    
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

    return schedule,obj,lldr_cpu_time


def DRO_seq_separate(N,mu,sigma,x_given,i,x_i_minus_1,t0_minus):
#    N = 3
#    mu = np.asarray([20,10,30,5,15,25])
#    sigma = np.asarray([0.1,0.1,0.1,1,1,1])
    
    sigma_square = sigma * sigma

    m = gp.Model("liftedLdr")
    
    
    t0 = m.addVar(vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t0')
    t1 = m.addVars(1,2*N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t1')
    t2 = m.addVars(1,2*N, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='t2')
    

    a = m.addVars(2*N, 1, vtype = GRB.CONTINUOUS,lb = 0,name='a')
    c = m.addVars(2*N, 1, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='c')
    
    d = m.addVars(2*N, 1, vtype = GRB.CONTINUOUS,lb = 0,name='d')
    f = m.addVars(2*N, 1, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name='f')
    
    x = m.addVars(N, vtype = GRB.BINARY,name='x')   
    # x = m.addVars(N, N, vtype = GRB.CONTINUOUS,lb = 0,name='x') # relax binary constraint
    
    obj = 0
    obj = obj + quicksum([t1[i,j] * mu[j] + t2[i,j]*sigma_square[j] for j in range(2*N)])
    obj = obj + t0

    m.setObjective(obj, GRB.MINIMIZE)
    

    qc1 = {}
    qc2 = {}
    m.addConstr(t0 >= quicksum([a[j,i] - c[j,i] * mu[j] for j in range(2*N)]) )

    for j in range(2*N):
        if j < N:
            m.addConstr(c[j,i] == t1[i,j])
        else:
            m.addConstr(c[j,i] == t1[i,j] - x[j-N])
        name = 'qc1_' + str(i) + str(j)
        qc1[i,j] = m.addVars(3, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name=name)
        m.addConstr(qc1[i,j][0] == a[j,i] + t2[i,j] )
        m.addConstr(qc1[i,j][1] == a[j,i] - t2[i,j] )
        m.addConstr(qc1[i,j][2] == c[j,i])
        m.addConstr(qc1[i,j][2]*qc1[i,j][2] + qc1[i,j][1] * qc1[i,j][1] <= qc1[i,j][0]*qc1[i,j][0] )
        m.addConstr(qc1[i,j][0] >= 0) 


    if i == 0:
        m.addConstr(t0 >= quicksum([d[j,i] - f[j,i] * mu[j] for j in range(2*N)]) )

        for j in range(2*N):
            name = 'qc2_' + str(i) + str(j)
            qc2[i,j] = m.addVars(3, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name=name)
            if j < N:
                m.addConstr(f[j,i] == t1[i,j])
            else:
                m.addConstr(f[j,i] == t1[i,j])
            # m.addConstr(4 * t2[i,j] * d[j,i] >= f[j,i] * f[j,i])

            m.addConstr(qc2[i,j][0] == d[j,i] + t2[i,j] )
            m.addConstr(qc2[i,j][1] == d[j,i] - t2[i,j] )
            m.addConstr(qc2[i,j][2] == f[j,i])
            m.addConstr(qc2[i,j][2]*qc2[i,j][2] + qc2[i,j][1] * qc2[i,j][1] <= qc2[i,j][0]*qc2[i,j][0] )
            m.addConstr(qc2[i,j][0] >= 0) 
    else:
        # Note this constraint
        m.addConstr(t0 - t0[i-1] >= quicksum([d[j,i] - f[j,i] * mu[j]]) )

        for j in range(2*N):
            name = 'qc2_' + str(i) + str(j)
            qc2[i,j] = m.addVars(3, vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY,name=name)
            if j < N:
                # Note this constraints
                m.addConstr(f[j,i] == t1[i,j] - t1[i-1,j] - x[i-1,j])
            else:
                m.addConstr(f[j,i] == t1[i,j] - t1[i-1,j])

            m.addConstr(qc2[i,j][0] == d[j,i] + t2[i,j] - t2[i-1,j] )
            m.addConstr(qc2[i,j][1] == d[j,i] - t2[i,j] + t2[i-1,j] )
            m.addConstr(qc2[i,j][2] == f[j,i])
            m.addConstr(qc2[i,j][2]*qc2[i,j][2] + qc2[i,j][1] * qc2[i,j][1] <= qc2[i,j][0]*qc2[i,j][0] )
            m.addConstr(qc2[i,j][0] >= 0) 




    m.addConstr(quicksum(x[j] for j in range(N)) == 1)
    
    # for i in range(N):
    #     for j in range(N):
    #         x[i,j].Start = x_given[i,j]

    m.setParam('OutputFlag', 1)
    m.setParam('MIPGap',0.01)
    m.setParam('TimeLimit',600)
    start = time.time()
    m.optimize()
    end = time.time()
    lldr_cpu_time = end - start
    #m.write("IB.LP")
    # dv = m.getConstrByName("convexity").getAttr("Pi")


    t0_result = t0.x
    t1_result = np.zeros((N,2*N))
    for ind in range(1):
        for j in range(2*N):
            t1_result[ind,j] = t1[ind,j].x
    t2_result = np.zeros((N,2*N))
    for ind in range(1):
        for j in range(2*N):
            t2_result[ind,j] = t2[ind,j].x
    x_result = np.zeros(N)
    for j in range(N):
        x_result[j] = x[j].x

    return x_result