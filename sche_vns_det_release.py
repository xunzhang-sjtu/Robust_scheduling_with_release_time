# 变邻域搜索解TSP问题
import numpy as np
import random as rd
import copy
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
# import vns_methods as vm
import time
import multiprocessing as mp


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
        x_matrix[i, int(np.round(solution[i]))-1] = 1
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


def compute_neiborhood_obj(neiborSolution,N,mu,sigma,r):
    obj_set = np.ones(N)*1000000000
    x_dict_set = {}

    N_sol = len(neiborSolution)
    num_cores = int(mp.cpu_count())
    # num_cores = 4

    len_sol = len(neiborSolution)
    pool = mp.Pool(num_cores) 
    result_it = []
    for j in range(len_sol):
        sol = neiborSolution[j]
        # solution = det_release_time_scheduling_given_seq(j,N,mu,sigma,r,sol)
        # obj_set[j] = solution['obj']

        result_it.append(pool.apply_async(det_release_time_scheduling_given_seq, args=(j,N,mu,sigma,r,sol,)))
    pool.close()
    pool.join()
    for j in range(len_sol):
        solution = result_it[j].get()
        obj_set[j] = solution['obj']
       
    #obj_set = np.sum(result_it_out,axis = 0)
    x_dict_set = neiborSolution
    return obj_set, x_dict_set


# def compute_neiborhood_obj(neiborSolution,N,mu,sigma,r):
#     obj_set = []
#     x_dict_set = {}

#     for j in range(len(neiborSolution)):
#         # obtain objective value
#         obj = det_release_time_scheduling_given_seq(N,mu,sigma,r,neiborSolution[j])
#         # obj = compute_obj(neiborSolution[j], N,mu_p,var_p,d,b,h)
#         obj_set.append(obj)
#         x_dict_set[j] = neiborSolution[j]
#         # print('neibors:', j,'obj:',obj)
#     return obj_set, x_dict_set


def variableNeighborhoodDescent(solution,N,mu,sigma,r):
    # obtain original objective
    sol_curr = det_release_time_scheduling_given_seq(0,N,mu,sigma,r,solution)
    curr_opt_obj = sol_curr['obj']
    it = 1

    i = 0
    dis, k = float("inf"), -1
    while i < 1:
        if i == 0:
            neiborSolution = neighborhoodOne(solution)
        elif i == 1:
            neiborSolution = neighborhoodTwo(solution)
        elif i == 2:
            neiborSolution = neighborhoodThree(solution)

        obj_set, x_dict_set = compute_neiborhood_obj(neiborSolution,N,mu,sigma,r)
        print('neigbor sol:',i,' iteration:',it,'opt obj:',np.min(obj_set))
        it = it + 1
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

def obtain_s_info(X,r_mu,r_sigma,p_mu,p_sigam):
    p_mu_x = X @ p_mu
    p_sigma_x = X @ p_sigam
    p_2mom_x = p_sigma_x * p_sigma_x + p_mu_x * p_mu_x        

    return p_mu_x,p_2mom_x


def vns(N,mu,sigma,r):

    # generate a initial solution
    currentSolution = np.arange(1,N+1)
    opt_x = currentSolution
    sol0 = det_release_time_scheduling_given_seq(0,N,mu,sigma,r,opt_x)
    opt_obj = sol0['obj'] # opt_obj = compute_obj(currentSolution,N,mu_p,var_p,d,b,h)

    gap = 100000000
    time_flag = 0
    iterx, iterxMax = 0, 1
    start_time = time.time()
    while iterx < iterxMax:
        # neighborhood of shaking
        shaking_neibors = shaking(currentSolution)
        [L1,L2] = np.shape(shaking_neibors)
        L1 = 1
        for k in range(L1):
            x_0 = shaking_neibors[k]
            des_opt_obj, des_Solution = variableNeighborhoodDescent(x_0,N,mu,sigma,r)
            if des_opt_obj < opt_obj:
                gap = opt_obj - des_opt_obj
                opt_obj = des_opt_obj
                opt_x = des_Solution
        
            end_time = time.time()
            time_gap = end_time - start_time
            # if time_gap > 600:
            #     time_flag = 1
            #     break
            
        # print('opt_obj:',opt_obj)
        # currentSolution = opt_x
        if time_flag == 1:
            break
        iterx += 1

    end_time = time.time()
    time_gap = end_time - start_time
    return opt_x,opt_obj,time_gap





def det_release_time_scheduling_given_seq(index,N,mu,sigma,r,x_seq):

    x,x_dict = decode(x_seq,N)
    mu = x @ mu
    sigma = x @ sigma
    r = x @ r
    s = np.zeros(N)
    for i in range(1,N):
        s[i] = r[i] - r[i-1]

    s[0] = r[0]

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0) # mute acedamic license info
        env.start()

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
        # model.Params.LogToConsole = 0

        model.optimize()    
        if model.status == 2 or model.status == 13:
            obj_val = model.getObjective().getValue()

        else:
            obj_val = 10000000

    # rho1_val = np.zeros(N)
    # rho2_val = np.zeros(N)
    # for i in range(N):
    #     rho1_val[i] = rho1[i].x
    #     rho2_val[i] = rho2[i].x
    # print('rho1:',rho1_val)
    # print('rho2:',rho2_val)
    sol_set = {}
    sol_set['index'] = index
    sol_set['obj'] = obj_val
    return sol_set