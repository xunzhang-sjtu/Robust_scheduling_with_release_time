
# **** variable neighborhood search ***************
# 变邻域搜索解TSP问题
import numpy as np
import copy
import time
import multiprocessing as mp
import time
from numba import jit

# import gurobipy as gp
# from gurobipy import GRB
# from gurobipy import quicksum
from mosek.fusion import *


def shaking(solution):

    L = len(solution)
    solution_shaking = []
    for i in range(1,L):
        sol = copy.deepcopy(solution)
        tem = sol[i]
        sol = np.delete(sol,i)
        sol = np.insert(sol,0,tem)
        solution_shaking.append(sol)

    return solution_shaking


def decode(solution):
    N = len(solution)
    x_matrix = np.zeros((N, N))
    for i in range(N):
        x_matrix[i, int(np.round(solution[i]))-1] = 1
    x_dict = {}
    for i in range(N):
        for j in range(N):
            x_dict[i+1, j+1] = x_matrix[i, j]
    return x_matrix,x_dict


def compute_neiborhood_obj(neiborSolution,N,r,c,M,p_hat,d_bar,model_mosek,models):
    
    x_dict_set = {}
    # start_time = time.time()
    len_sol = len(neiborSolution)
    obj_set = np.ones(len_sol)*1000000000

        # # ************** parallel computing ****************************
        # Set up n copies of the model with different data
    threadspermodel = 1    # Number of threads per each model
    threadpoolsize = N     # Total number of threads available
    for j in range(len_sol):

        sol = neiborSolution[j]
        x_matrix,x_dict = decode(sol+1)
        # obj_set[j] = det_release_time_scheduling_wass(N,r,c,M,p_hat,d_bar,x_matrix)

        xr_tem = x_matrix @ r
        xd_tem = x_matrix @ d_bar
        xp_tem = x_matrix @ p_hat


        models[j].getParameter("xr").setValue(xr_tem)
        models[j].getParameter("xd").setValue(xd_tem)
        models[j].getParameter("xp").setValue(xp_tem)
        models[j].getParameter("c").setValue(c)


        # We can set the number of threads individually per model
        models[j].setSolverParam("numThreads", threadspermodel)

    # Solve all models in parallel
    status = Model.solveBatch(False,         # No race
                                -1.0,          # No time limit
                                threadpoolsize,
                                models)        # Array of Models to solve

    # Access information about each model
    for j in range(len_sol):
        if status[j] == SolverStatus.OK:
            # print("Model {}: Status {}, Solution Status {}, Objective {:.3f}, Time {:.3f}".format(
            #     j, 
            #     status[j],
            #     models[j].getPrimalSolutionStatus(),
            #     models[j].primalObjValue(),
            #     models[j].getSolverDoubleInfo("optimizerTime")))
            primal_obj = models[j].primalObjValue(),
            obj_set[j] = primal_obj[0]
        else:
            print("Model {}: not solved".format(j))



        # obj_set[j] = np.min(obj_arr)

    
    x_dict_set = neiborSolution

    return obj_set, x_dict_set



def variableNeighborhoodDescent(N,r,c,M,p_hat,d_bar,solution,model_mosek,models,sol_saa):
    # obtain original objective
    x_matrix,x_dict = decode(solution+1)

    x_matrix,x_dict = decode(sol_saa['seq'])
    solution = sol_saa['seq'] - 1

    xr_tem = x_matrix @ r
    xd_tem = x_matrix @ d_bar
    xp_tem = x_matrix @ p_hat

    model_mosek.getParameter("xr").setValue(xr_tem)
    model_mosek.getParameter("xd").setValue(xd_tem)
    model_mosek.getParameter("xp").setValue(xp_tem)
    model_mosek.getParameter("c").setValue(c)

    model_mosek.solve()
    primal_obj = model_mosek.primalObjValue()
    # sol = theta.level()
    # ka = model_mosek.getVariable('ka').level()
    curr_opt_obj = primal_obj



    it = 1
    i = 0
    while i < 1:
        if i == 0:
            neiborSolution = neighborhoodOne(solution)
        elif i == 1:
            # neiborSolution = neighborhoodTwo(solution)
            neiborSolution = shaking(solution)
        elif i == 2:
            neiborSolution = neighborhoodThree(solution)

        obj_set, x_dict_set = compute_neiborhood_obj(neiborSolution,N,r,c,M,p_hat,d_bar,model_mosek,models)
        it = it + 1
        # obtain the index of minimum objective value
        index_min = np.argmin(obj_set)
        if obj_set[index_min] < curr_opt_obj:
            solution = x_dict_set[index_min] # update solution
            curr_opt_obj = obj_set[index_min]  # update current value
            i = 0            

        else:
            i += 1
        # print('iteration=',it,'obj:',curr_opt_obj,' seq:',solution)

    sol = {}
    sol['obj'] = curr_opt_obj
    sol['sol'] = solution
    return sol


def neighborhoodOne(sol):  # swap算子
    neighbor = []

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

    return neighbor


def neighborhoodThree(sol):  # two_h_opt_swap算子
    neighbor = []
    for j in range(len(sol)-3):
        s = copy.deepcopy(sol)
        x = s[j+3]
        s[j+3] = s[j]
        s[j] = x
        neighbor.append(s)

    return neighbor


def vns(N,r,c,M,p_hat,d_bar,model_mosek,models,sol_saa):



    # generate a initial solution
    currentSolution = np.arange(0,N)
    opt_x = currentSolution
    opt_obj = 1000000000000000

    time_flag = 0
    iterx, iterxMax = 0, 1
    start_time = time.time()
    while iterx < iterxMax:
        # neighborhood of shaking
        shaking_neibors = shaking(currentSolution)
        [L1,L2] = np.shape(shaking_neibors)
        L1 = 1
        for k in range(L1):
            seq_0 = shaking_neibors[k]
            sol = variableNeighborhoodDescent(N,r,c,M,p_hat,d_bar,seq_0,model_mosek,models,sol_saa)
            # sol = parallel_compute_ka(N,r,c,M,p_hat,d_bar,seq_0)
            des_opt_obj = sol['obj']
            des_Solution = sol['sol']

            if des_opt_obj < opt_obj:
                opt_obj = des_opt_obj
                opt_x = des_Solution
            # print('opt_obj:',opt_obj,'seq',opt_x)

            end_time = time.time()
            time_gap = end_time - start_time

        currentSolution = opt_x
        if time_flag == 1:
            break
        iterx += 1

    end_time = time.time()
    time_gap = end_time - start_time
    sol = {}
    sol['c'] = c
    sol['time'] = time_gap
    sol['x_seq'] = opt_x
    sol['obj'] = opt_obj

    return sol




