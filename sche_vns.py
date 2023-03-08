# 变邻域搜索解TSP问题
import numpy as np
import random as rd
import copy
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
# import vns_methods as vm
import time


def shaking(solution):
    # solutioni = []
    # for i in range(4):
    #     lis = solution[i:i+3]
    #     solutioni.append(lis)
    # sequence = [i for i in range(4)]
    # rd.shuffle(sequence)
    # solution = []
    # for se in sequence:
    #     solution += solutioni[se]
    solution_shaking = []
    for i in range(1,len(solution)):
        sol = copy.deepcopy(solution)
        tem = sol[i]
        sol = np.delete(sol,i)
        sol = np.insert(sol,0,tem)
        solution_shaking.append(sol)

    return solution_shaking


def decode(solution, N):
    x_matrix = np.zeros((N, N))
    for i in range(N):
        x_matrix[i, int(solution[i])-1] = 1
    x_dict = {}
    for i in range(N):
        for j in range(N):
            x_dict[i+1, j+1] = x_matrix[i, j]
    return x_matrix,x_dict


# def compute_obj(solution, N, mu, sigma_square, d, b, h):
#     mu_i = 0
#     sigma_i = 0
#     obj = 0
#     for i in range(N):
#         job_index = solution[i] - 1
#         mu_i = mu_i + mu[job_index]
#         mu_i_d = mu_i - d[job_index]
#         sigma_i = sigma_i + sigma_square[job_index]
#         obj_i = (h+b)*(0.5 * mu_i_d + 0.5 * np.sqrt(mu_i_d**2 + sigma_i)) + h*(d[job_index] - mu_i)
#         obj = obj + obj_i

#     return obj


def compute_neiborhood_obj(neiborSolution, N,r_hat,p_hat):
    obj_set = []
    x_dict_set = {}
    for j in range(len(neiborSolution)):
        # obtain objective value
        obj = rand_releas_time_scheduling_given_X(neiborSolution[j], N,r_hat,p_hat)
        # obj = compute_obj(neiborSolution[j], N,mu_p,var_p,d,b,h)
        obj_set.append(obj)
        x_dict_set[j] = neiborSolution[j]
        # print('neibors:', j,'obj:',obj)
    return obj_set, x_dict_set


def variableNeighborhoodDescent(solution,N,r_hat,p_hat):
    # obtain original objective
    curr_opt_obj = rand_releas_time_scheduling_given_X(solution,N,r_hat,p_hat)

    i = 0
    dis, k = float("inf"), -1
    while i < 2:
        if i == 0:
            neiborSolution = neighborhoodOne(solution)
        elif i == 1:
            neiborSolution = neighborhoodTwo(solution)
        elif i == 2:
            neiborSolution = neighborhoodThree(solution)

        obj_set, x_dict_set = compute_neiborhood_obj(neiborSolution,N,r_hat,p_hat)
        # obtain the index of minimum objective value
        index_min = np.argmin(obj_set)
        if obj_set[index_min] < curr_opt_obj:
            solution = x_dict_set[index_min] # update solution
            curr_opt_obj = obj_set[index_min]  # update current value
            i = 0            
        # for j in range(len(neiborSolution)):
        #     if obj_set[j] < dis:
        #         dis = obj_set[j]
        #         k = j
        # if dis < curr_opt_obj:
        #     solution = neiborSolution[k]
        #     curr_opt_obj = dis
        #     i = 0
        else:
            i += 1
    return curr_opt_obj, solution


def neighborhoodOne(sol):  # swap算子
    neighbor = []
#    for i in range(len(sol)):
#        for j in range(i+1,len(sol)):
#            s = copy.deepcopy(sol)
#            x = s[j]
#            s[j] = s[i]
#            s[i] = x
#            neighbor.append(s)

    for j in range(len(sol)-1):
        s = copy.deepcopy(sol)
        x = s[j+1]
        s[j+1] = s[j]
        s[j] = x
        neighbor.append(s)

    return neighbor


def neighborhoodTwo(sol):  # two_opt_swap算子
    neighbor = []

    for j in range(len(sol)-2):
        s = copy.deepcopy(sol)
        x = s[j+2]
        s[j+2] = s[j]
        s[j] = x
        neighbor.append(s)


    # for i in range(len(sol)):
    #     for j in range(i + 3, len(sol)):  # 这里j从i+3开始是为了不产生跟swap算子重复的解
    #         s = copy.deepcopy(sol)
    #         s1 = s[i:j+1]
    #         s1.reverse()
    #         s = s[:i] + s1 + s[j+1:]
    #         neighbor.append(s)
    return neighbor


def neighborhoodThree(sol):  # two_h_opt_swap算子
    neighbor = []
    for i in range(len(sol)):
        for j in range(i+1, len(sol)):
            s = copy.deepcopy(sol)
            s = [s[i]] + [s[j]] + s[:i] + s[i+1:j] + s[j+1:]
            neighbor.append(s)
    return neighbor

def obtain_s_info(X,r_hat,p_hat,n):
    p_hat_x = X @ p_hat
    p_mu_x = np.mean(p_hat_x,axis = 1)
    p_2mom_x = np.var(p_hat_x,axis = 1) + p_mu_x * p_mu_x        

    # Y = X
    # Y[1:n,:] = Y[1:n,:] - Y[0:n-1,:]
    # s_hat = Y @ r_hat
    # s_mu_x1 = np.mean(s_hat,axis = 1)
    # s_2Mom_x1 = np.var(s_hat,axis = 1) + s_mu_x1 * s_mu_x1

    r_hat_x = X @ r_hat
    r_mu_x = np.mean(r_hat_x,axis = 1)
    r_sigma_x = np.var(r_hat,axis = 1)

    s_mu_x = np.zeros(n)
    s_2Mom_x = np.zeros(n)
    s_mu_x[0] = r_mu_x[0]
    s_2Mom_x[0] = r_sigma_x[0] + r_mu_x[0] * r_mu_x[0]
    for i in range(1,n):
        s_mu_x[i] = r_mu_x[i] - r_mu_x[i-1]
        s_2Mom_x[i] = r_sigma_x[i] + r_sigma_x[i-1] + (r_mu_x[i] - r_mu_x[i-1])*(r_mu_x[i] - r_mu_x[i-1])

    return p_mu_x,p_2mom_x,s_mu_x,s_2Mom_x


def vns(N,r_hat,p_hat):

    # generate a initial solution
    currentSolution = np.arange(1,N+1)
    opt_x = currentSolution
    opt_obj = rand_releas_time_scheduling_given_X(opt_x,N,r_hat,p_hat)
    # opt_obj = compute_obj(currentSolution,N,mu_p,var_p,d,b,h)

    gap = 100000000
    time_flag = 0
    iterx, iterxMax = 0, 1
    start_time = time.time()
    while iterx < iterxMax:
        # neighborhood of shaking
        shaking_neibors = shaking(currentSolution)
        [L1,L2] = np.shape(shaking_neibors)
        for k in range(L1):
            x_0 = shaking_neibors[k]
            des_opt_obj, des_Solution = variableNeighborhoodDescent(x_0,N,r_hat,p_hat)
            if des_opt_obj < opt_obj:
                gap = opt_obj - des_opt_obj
                opt_obj = des_opt_obj
                opt_x = des_Solution
        
            end_time = time.time()
            time_gap = end_time - start_time
            if time_gap > 600:
                time_flag = 1
                break
            
        # print('opt_obj:',opt_obj)
        # currentSolution = opt_x
        if time_flag == 1:
            break
        iterx += 1

    end_time = time.time()
    time_gap = end_time - start_time
    return opt_x,opt_obj,time_gap



# h = 1
# b = 2

# N = 4 # number of jobs
# mu_p = np.random.uniform(1,10,N) # mean processing time
# # mu_p = np.ones(N)
# var_p = 10 * np.random.uniform(1,5,N) # variance of process time
# d = np.random.uniform(1,10*N,N) # due date



# if __name__ == '__main__':

#     # generate a initial solution
#     currentSolution = np.arange(1, 1+N)
#     opt_x = currentSolution
#     # obtain initial value 
#     opt_obj = compute_obj(currentSolution,N,mu_p,var_p,d,b,h)


#     iterx, iterxMax = 0, 5
#     while iterx < iterxMax:
#         # neighborhood of shaking
#         shaking_neibors = shaking(currentSolution)
#         [L1,L2] = np.shape(shaking_neibors)
#         for k in range(L1):
#             x_0 = shaking_neibors[k]
#             des_opt_obj, des_Solution = variableNeighborhoodDescent(x_0,N,mu_p,var_p,d,b,h)
#             if des_opt_obj < opt_obj:
#                 opt_obj = des_opt_obj
#                 opt_x = des_Solution

#         print('opt_obj:',opt_obj)
#         currentSolution = opt_x
#         iterx += 1
#     print(opt_obj)
#     print(opt_x)



def rand_releas_time_scheduling_given_X(sol,N,r_hat,p_hat):
    x_matrix,x_dict = decode(sol,N)
    mu_p,sigma_p,mu_r,sigma_r = obtain_s_info(x_matrix,r_hat,p_hat,N)

    # mu_p = np.ones(N)
    # sigma_p = np.ones(N) + np.ones(N) * 0.001
    # mu_r = np.asarray([1,3,5])
    # sigma_r = mu_r*mu_r + np.ones(N) * 0.001






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




    model.setParam('OutputFlag', 0)
    # model.setParam('TimeLimit',600)
    # model.write("E:\\onedrive\\dro.LP")
   
    model.optimize()    
    if model.status == 2 or model.status == 13:
        obj_val = model.getObjective().getValue()
        x_tem = np.zeros((N,N))
        # x_seq = x_tem @ np.arange(N)
    else:
        obj_val = -1000
        # x_seq = np.zeros(N)   
    
    return obj_val


def det_release_time_scheduling_given_seq(N,mu,sigma,r,sol):
    x_matrix,x_dict = decode(sol,N)
    mu_p,sigma_p,mu_r,sigma_r = obtain_s_info(x_matrix,r_hat,p_hat,N)


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
                print('k:',k,',j:',j,',i:',i)
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
                model.addConstr(qc[i,j][2] == (j-i) - rho1[i])
                model.addConstr(qc[i,j][1] == rho2[i] - xi[i,j])
                model.addConstr(qc[i,j][0] == rho2[i] + xi[i,j])
                model.addConstr(qc[i,j][0] >= 0)
                model.addConstr(qc[i,j][2] * qc[i,j][2] +  qc[i,j][1] * qc[i,j][1] <= qc[i,j][0] * qc[i,j][0])
            else:
                model.addConstr(qc[i,j][2] == (j-i) - rho1[i])
                model.addConstr(qc[i,j][1] == rho2[i] - xi[i,j])
                model.addConstr(qc[i,j][0] == rho2[i] + xi[i,j])
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

    model.setParam('OutputFlag', 1)
    # model.setParam('TimeLimit',600)
    model.write("E:\\onedrive\\dro.LP")
   
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
    
    return obj_val, x_seq