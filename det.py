"""
Created on Thu Aug 27 18:31:55 2020

@author: xunzhang
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum

import time
def det_seq(N,r,p):
    m = gp.Model("Det")
    t = m.addVars(N,vtype = GRB.CONTINUOUS,lb = -GRB.INFINITY, name='t')
    x = m.addVars(N, N, vtype = GRB.BINARY,name='x')    

    obj = quicksum(t) + quicksum(p)
    m.setObjective(obj, GRB.MINIMIZE )
        
    for i in range(1,N):
        m.addConstr(t[i] >= t[i-1] + quicksum([x[i-1,j] * p[j] for j in range(N)]))
    
    for i in range(N):
        m.addConstr(t[i] >= quicksum([x[i,j] * r[j] for j in range(N)]))

        # At most one queen per row
        m.addConstr(quicksum([x[i,j] for j in range(N)]) == 1)
        # At most one queen per column
        m.addConstr(quicksum([x[j,i] for j in range(N)]) == 1)
    
    
    m.setParam('OutputFlag', 0)
    # m.setParam('MIPGap',0.01)
    # m.setParam('TimeLimit',600)
    startTime = time.time()
    m.optimize()
    endTime = time.time()
    detTime = endTime - startTime
    #m.write("IB.LP")
    
    
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

    return schedule,obj,detTime
    