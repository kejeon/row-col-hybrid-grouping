import numpy as np
import gurobipy as gp
from gurobipy import GRB
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.setParam('LogToConsole', 0)
env.setParam('Threads', 1)
env.start()


class gurobi_free_1():
    def __init__(self, C, R, mem_q_lvl, R_start, rank_vec, max_q_code, solver=None):
        super().__init__()
        self.rank_vec = rank_vec
        self.cumsum_mat = np.triu(np.ones((R, R)))[:, R_start:]
        self.mem_q_lvl = mem_q_lvl
        self.C = C
        self.R = R
        self.R_start = R_start

        self.model = gp.Model("milp")
        self.x_pos = self.model.addMVar((C, R), lb=0, ub=mem_q_lvl-1,
                                    vtype=GRB.INTEGER, name="x_pos")
        self.x_neg = self.model.addMVar((C, R), lb=0, ub=mem_q_lvl-1,
                                    vtype=GRB.INTEGER, name="x_neg")

        self.model.Params.Threads = 1
        self.model.Params.OutputFlag = 0
        self.model.Params.LogToConsole = 0 

    def cvx_solve(self, all_fault_pos, all_fault_neg, saf0_pos, saf0_neg, q_code):
        self.model.reset()
        self.model.remove(self.model.getConstrs())

        self.cvx_q_code = q_code
        self.cvx_all_fault_pos = np.logical_not(all_fault_pos)
        self.cvx_all_fault_neg = np.logical_not(all_fault_neg)
        self.cvx_saf0_pos = saf0_pos
        self.cvx_saf0_neg = saf0_neg

        self.x_pos_corrupt = self.cvx_all_fault_pos*self.x_pos + self.cvx_saf0_pos*(self.mem_q_lvl-1)
        self.x_neg_corrupt = self.cvx_all_fault_neg*self.x_neg + self.cvx_saf0_neg*(self.mem_q_lvl-1)
        self.x_corrupt = self.x_pos_corrupt - self.x_neg_corrupt
        self.q_code_corrupt = self.rank_vec @ self.x_corrupt @ self.cumsum_mat
        self.q_code_residual = self.q_code_corrupt - self.cvx_q_code

        self.t = self.model.addMVar((self.R-self.R_start,), lb=0, vtype=GRB.CONTINUOUS, name="t")

        self.model.setObjective(gp.quicksum(self.t), GRB.MINIMIZE)
        self.model.addConstr(self.t >= 0)
        self.model.addConstr(self.t >= self.q_code_residual)
        self.model.addConstr(self.t >= -self.q_code_residual)

        self.model.optimize()

        # get the values
        rc_pos = self.x_pos.X
        rc_neg = self.x_neg.X
        rc_pair = np.stack([rc_pos, rc_neg], axis=0)
        return rc_pair

    def cvx_solve_for_list(self, all_faults_list, saf0_list, q_code_list):
        rc_pair_list = np.empty_like(all_faults_list, dtype=int)
        num_q_codes = len(q_code_list)

        for i in range(num_q_codes):
            rc_pair = self.cvx_solve(all_faults_list[i][0],
                                    all_faults_list[i][1],
                                    saf0_list[i][0],
                                    saf0_list[i][1],
                                    q_code_list[i])
            if rc_pair[0] is None:
                print(f"Failed to solve for q_code {i}")
                continue
            rc_pair_list[i] = rc_pair        
        
        return rc_pair_list

class gurobi_free_2():
    def __init__(self, C, R, mem_q_lvl, R_start, rank_vec, max_q_code, solver=None):
        super().__init__()
        self.rank_vec = rank_vec
        self.cumsum_mat = np.triu(np.ones((R, R)))[:, R_start:]
        self.mem_q_lvl = mem_q_lvl
        self.C = C
        self.R = R
        self.R_start = R_start

        self.model = gp.Model("milp")
        self.x_pos = self.model.addMVar((C, R), lb=0, ub=mem_q_lvl-1,
                                    vtype=GRB.INTEGER, name="x_pos")
        self.x_neg = self.model.addMVar((C, R), lb=0, ub=mem_q_lvl-1,
                                    vtype=GRB.INTEGER, name="x_neg")
        self.cvx_all_fault_pos = self.model.addMVar((C, R), lb=0, ub=1,
                                    vtype=GRB.BINARY, name="all_fault_pos")
        self.cvx_all_fault_neg = self.model.addMVar((C, R), lb=0, ub=1,
                                    vtype=GRB.BINARY, name="all_fault_neg")
        self.cvx_saf0_pos = self.model.addMVar((C, R), lb=0, ub=1,
                                    vtype=GRB.BINARY, name="saf0_pos")
        self.cvx_saf0_neg = self.model.addMVar((C, R), lb=0, ub=1,
                                    vtype=GRB.BINARY, name="saf0_neg")
        self.cvx_q_code = self.model.addMVar((R-R_start,), lb=0, ub=max_q_code,
                                    vtype=GRB.INTEGER, name="q_code")

        self.x_pos_corrupt = self.cvx_all_fault_pos*self.x_pos + self.cvx_saf0_pos*(self.mem_q_lvl-1)
        self.x_neg_corrupt = self.cvx_all_fault_neg*self.x_neg + self.cvx_saf0_neg*(self.mem_q_lvl-1)
        self.x_corrupt = self.x_pos_corrupt - self.x_neg_corrupt
        self.q_code_corrupt = self.rank_vec @ self.x_corrupt @ self.cumsum_mat
        self.q_code_residual = self.q_code_corrupt - self.cvx_q_code

        self.t = self.model.addMVar((R-R_start,), lb=0, vtype=GRB.CONTINUOUS, name="t")

        self.model.setObjective(gp.quicksum(self.t), GRB.MINIMIZE)
        self.model.addConstr(self.t >= 0)
        self.model.addConstr(self.t >= self.q_code_residual)
        self.model.addConstr(self.t >= -self.q_code_residual)
        
        self.model.Params.Threads = 1
        self.model.Params.OutputFlag = 0
        self.model.Params.LogToConsole = 0 


    def cvx_solve(self, all_fault_pos, all_fault_neg, saf0_pos, saf0_neg, q_code):
        self.model.reset()

        # make cvx_q_code to q_code and fix its values
        self.cvx_q_code.UB = q_code
        self.cvx_q_code.LB = q_code
        self.cvx_all_fault_pos.UB = all_fault_pos
        self.cvx_all_fault_pos.LB = all_fault_pos
        self.cvx_all_fault_neg.UB = all_fault_neg
        self.cvx_all_fault_neg.LB = all_fault_neg
        self.cvx_saf0_pos.UB = saf0_pos
        self.cvx_saf0_pos.LB = saf0_pos
        self.cvx_saf0_neg.UB = saf0_neg
        self.cvx_saf0_neg.LB = saf0_neg

        self.model.optimize()

        # get the values
        rc_pos = self.x_pos.X
        rc_neg = self.x_neg.X
        rc_pair = np.stack([rc_pos, rc_neg], axis=0)
        return rc_pair

    def cvx_solve_for_list(self, all_faults_list, saf0_list, q_code_list):
        rc_pair_list = np.empty_like(all_faults_list, dtype=int)
        num_q_codes = len(q_code_list)
        all_faults_list = np.logical_not(all_faults_list)

        for i in range(num_q_codes):
            rc_pair = self.cvx_solve(all_faults_list[i][0],
                                    all_faults_list[i][1],
                                    saf0_list[i][0],
                                    saf0_list[i][1],
                                    q_code_list[i])
            if rc_pair[0] is None:
                print(f"Failed to solve for q_code {i}")
                continue
            rc_pair_list[i] = rc_pair        
        
        return rc_pair_list

    def tune_model(self):
        self.model.tune()
        for i in range(self.model.tuneResultCount):
            self.model.getTuneResult(i)
            self.model.write('tune'+str(i)+'.prm')



class gurobi_free_3():
    def __init__(self, C, R, mem_q_lvl, R_start, rank_vec, max_q_code, solver=None):
        super().__init__()
        self.rank_vec = rank_vec
        self.cumsum_mat = np.triu(np.ones((R, R)))[:, R_start:]
        self.mem_q_lvl = mem_q_lvl
        self.C = C
        self.R = R
        self.R_start = R_start

        self.model = gp.Model("milp")
        self.x = self.model.addMVar((C, R), lb=-mem_q_lvl+1, ub=mem_q_lvl-1,
                                    vtype=GRB.INTEGER, name="x")
        self.cvx_saf0_pos = self.model.addMVar((C, R), lb=0, ub=1,
                                    vtype=GRB.BINARY, name="saf0_pos")
        self.cvx_saf0_neg = self.model.addMVar((C, R), lb=0, ub=1,
                                    vtype=GRB.BINARY, name="saf0_neg")
        self.cvx_q_code = self.model.addMVar((R-R_start,), lb=0, ub=max_q_code,
                                    vtype=GRB.INTEGER, name="q_code")

        self.x_pos_corrupt = self.cvx_saf0_pos*(self.mem_q_lvl-1)
        self.x_neg_corrupt = self.cvx_saf0_neg*(self.mem_q_lvl-1)
        self.x_corrupt = self.x + self.x_pos_corrupt - self.x_neg_corrupt
        self.q_code_corrupt = self.rank_vec @ self.x_corrupt @ self.cumsum_mat
        self.q_code_residual = self.q_code_corrupt - self.cvx_q_code

        self.t = self.model.addMVar((R-R_start,), lb=0, vtype=GRB.CONTINUOUS, name="t")

        self.model.setObjective(gp.quicksum(self.t), GRB.MINIMIZE)
        self.model.addConstr(self.t >= 0)
        self.model.addConstr(self.t >= self.q_code_residual)
        self.model.addConstr(self.t >= -self.q_code_residual)
        
        self.model.Params.Threads = 1
        self.model.Params.OutputFlag = 0
        self.model.Params.LogToConsole = 0
        self.model.Params.LogFile = "gurobi-3.log" 


    def cvx_solve(self, all_fault_pos, all_fault_neg, saf0_pos, saf0_neg, q_code):
        self.model.reset()

        # make cvx_q_code to q_code and fix its values
        self.cvx_q_code.UB = q_code
        self.cvx_q_code.LB = q_code
        self.cvx_saf0_pos.UB = saf0_pos
        self.cvx_saf0_pos.LB = saf0_pos
        self.cvx_saf0_neg.UB = saf0_neg
        self.cvx_saf0_neg.LB = saf0_neg

        both_no_fault = all_fault_pos * all_fault_neg
        only_pos_fault = all_fault_pos * (~all_fault_neg)
        only_neg_fault = (~all_fault_pos) * all_fault_neg

        self.x.UB = both_no_fault + only_pos_fault
        self.x.LB = -(both_no_fault + only_neg_fault).astype(int)
        self.model.optimize()

        # get the values
        rc = self.x.X
        # set negative values to 0 and return as rc_pos
        rc_pos = np.round(np.maximum(rc, 0))
        rc_neg = np.round(np.maximum(-rc, 0))
        rc_pair = np.stack([rc_pos, rc_neg], axis=0)
        return rc_pair
    
    def _cvx_solve(self, x_ub, x_lb, saf0_pos, saf0_neg, q_code):
        self.model.reset()

        # make cvx_q_code to q_code and fix its values
        self.cvx_q_code.UB = q_code
        self.cvx_q_code.LB = q_code
        self.cvx_saf0_pos.UB = saf0_pos
        self.cvx_saf0_pos.LB = saf0_pos
        self.cvx_saf0_neg.UB = saf0_neg
        self.cvx_saf0_neg.LB = saf0_neg

        self.x.UB = x_ub
        self.x.LB = x_lb
        self.model.optimize()

        # get the values
        rc = self.x.X
        return rc

    def cvx_solve_for_list(self, all_faults_list, saf0_list, q_code_list):
        rc_pair_list = np.empty_like(all_faults_list[:, 0], dtype=int)
        num_q_codes = len(q_code_list)
        all_faults_list = np.logical_not(all_faults_list)

        all_fault_pos = all_faults_list[:, 0]
        all_fault_neg = all_faults_list[:, 1]
        saf0_pos = saf0_list[:, 0]
        saf0_neg = saf0_list[:, 1]

        both_no_fault = all_fault_pos & all_fault_neg
        only_pos_fault = all_fault_pos & (~all_fault_neg)
        only_neg_fault = (~all_fault_pos) & all_fault_neg

        x_ub = (both_no_fault | only_pos_fault).astype(int) * (self.mem_q_lvl-1)
        x_lb = -(both_no_fault | only_neg_fault).astype(int) * (self.mem_q_lvl-1)

        for i in range(num_q_codes):
            # rc_pair = self.cvx_solve(all_faults_list[i][0],
            #                         all_faults_list[i][1],
            #                         saf0_list[i][0],
            #                         saf0_list[i][1],
            #                         q_code_list[i])
            rc_pair = self._cvx_solve(x_ub[i], x_lb[i], 
                                      saf0_pos[i], saf0_neg[i], 
                                      q_code_list[i])
            if rc_pair[0] is None:
                print(f"Failed to solve for q_code {i}")
                continue
            rc_pair_list[i] = rc_pair        
        
        rc_pos = np.round(np.maximum(rc_pair_list, 0))
        rc_neg = np.round(np.maximum(-rc_pair_list, 0))
        rc_pair_list = np.stack([rc_pos, rc_neg], axis=1)

        return rc_pair_list
    
    def tune_model(self):
        self.model.tune()
        for i in range(self.model.tuneResultCount):
            self.model.getTuneResult(i)
            self.model.write('tune'+str(i)+'.prm')

class gurobi_free_4():
    def __init__(self, C, R, mem_q_lvl, R_start, rank_vec, max_q_code, solver=None):
        super().__init__()
        self.rank_vec = rank_vec
        self.cumsum_mat = np.triu(np.ones((R, R)))[:, R_start:]
        self.mem_q_lvl = mem_q_lvl
        self.C = C
        self.R = R
        self.R_start = R_start

        self.model = gp.Model("milp", env=env)        
        self.x = self.model.addMVar((C, R), lb=-mem_q_lvl+1, ub=mem_q_lvl-1,
                                    vtype=GRB.INTEGER, name="x")
        self.cvx_saf0_pos = self.model.addMVar((C, R), lb=0, ub=1,
                                    vtype=GRB.BINARY, name="saf0_pos")
        self.cvx_saf0_neg = self.model.addMVar((C, R), lb=0, ub=1,
                                    vtype=GRB.BINARY, name="saf0_neg")
        self.cvx_q_code = self.model.addMVar((R-R_start,), lb=0, ub=max_q_code,
                                    vtype=GRB.INTEGER, name="q_code")

        self.x_pos_corrupt = self.cvx_saf0_pos*(self.mem_q_lvl-1)
        self.x_neg_corrupt = self.cvx_saf0_neg*(self.mem_q_lvl-1)
        self.x_corrupt = self.x + self.x_pos_corrupt - self.x_neg_corrupt
        self.q_code_corrupt = self.rank_vec @ self.x_corrupt @ self.cumsum_mat
        self.q_code_residual = (self.q_code_corrupt - self.cvx_q_code)/max_q_code

        self.t = self.model.addMVar((R-R_start,), lb=0, vtype=GRB.CONTINUOUS, name="t")

        self.model.setObjective(gp.quicksum(self.t), GRB.MINIMIZE)
        self.model.addConstr(self.t >= 0)
        self.model.addConstr(self.t >= self.q_code_residual)
        self.model.addConstr(self.t >= -self.q_code_residual)

        self.model.Params.Threads = 1
        self.model.Params.OutputFlag = 0
        self.model.Params.LogToConsole = 0
        # self.model.Params.LogFile = "gurobi-4.log" 

    def cvx_solve(self, all_fault_pos, all_fault_neg, saf0_pos, saf0_neg, q_code):
        self.model.reset()

        # make cvx_q_code to q_code and fix its values
        self.cvx_q_code.UB = q_code
        self.cvx_q_code.LB = q_code
        self.cvx_saf0_pos.UB = saf0_pos
        self.cvx_saf0_pos.LB = saf0_pos
        self.cvx_saf0_neg.UB = saf0_neg
        self.cvx_saf0_neg.LB = saf0_neg

        both_no_fault = all_fault_pos & all_fault_neg
        only_pos_fault = all_fault_pos & (~all_fault_neg)
        only_neg_fault = (~all_fault_pos) & all_fault_neg

        self.x.UB = (both_no_fault | only_pos_fault).astype(int) * (self.mem_q_lvl-1)
        self.x.LB = -(both_no_fault | only_neg_fault).astype(int) * (self.mem_q_lvl-1)
        self.model.optimize()

        # get the values
        rc = self.x.X
        # set negative values to 0 and return as rc_pos
        rc_pos = np.round(np.maximum(rc, 0))
        rc_neg = np.round(np.maximum(-rc, 0))
        rc_pair = np.stack([rc_pos, rc_neg], axis=0)
        return rc_pair
    
    def _cvx_solve(self, x_ub, x_lb, saf0_pos, saf0_neg, q_code):
        self.model.reset()

        # make cvx_q_code to q_code and fix its values
        self.cvx_q_code.UB = q_code
        self.cvx_q_code.LB = q_code
        self.cvx_saf0_pos.UB = saf0_pos
        self.cvx_saf0_pos.LB = saf0_pos
        self.cvx_saf0_neg.UB = saf0_neg
        self.cvx_saf0_neg.LB = saf0_neg

        self.x.UB = x_ub
        self.x.LB = x_lb
        self.model.optimize()

        # get the values
        rc = self.x.X
        return rc

    def cvx_solve_for_list(self, all_faults_list, saf0_list, q_code_list):
        rc_pair_list = np.empty_like(all_faults_list[:, 0], dtype=int)
        num_q_codes = len(q_code_list)
        all_faults_list = np.logical_not(all_faults_list)

        all_fault_pos = all_faults_list[:, 0]
        all_fault_neg = all_faults_list[:, 1]
        saf0_pos = saf0_list[:, 0]
        saf0_neg = saf0_list[:, 1]

        both_no_fault = all_fault_pos & all_fault_neg
        only_pos_fault = all_fault_pos & (~all_fault_neg)
        only_neg_fault = (~all_fault_pos) & all_fault_neg

        x_ub = (both_no_fault | only_pos_fault).astype(int) * (self.mem_q_lvl-1)
        x_lb = -(both_no_fault | only_neg_fault).astype(int) * (self.mem_q_lvl-1)

        for i in range(num_q_codes):
            # rc_pair = self.cvx_solve(all_faults_list[i][0],
            #                         all_faults_list[i][1],
            #                         saf0_list[i][0],
            #                         saf0_list[i][1],
            #                         q_code_list[i])
            rc_pair = self._cvx_solve(x_ub[i], x_lb[i], 
                                      saf0_pos[i], saf0_neg[i], 
                                      q_code_list[i])
            if rc_pair[0] is None:
                print(f"Failed to solve for q_code {i}")
                continue
            rc_pair_list[i] = rc_pair        
        
        rc_pos = np.round(np.maximum(rc_pair_list, 0))
        rc_neg = np.round(np.maximum(-rc_pair_list, 0))
        rc_pair_list = np.stack([rc_pos, rc_neg], axis=1)

        return rc_pair_list

class GurobiFree:
    def __init__(self, C, R, mem_q_lvl, R_start, rank_vec, max_q_code, solver=None):
        super().__init__()
        self.rank_vec = rank_vec
        self.cumsum_mat = np.triu(np.ones((R, R)))[:, R_start:]
        self.mem_q_lvl = mem_q_lvl
        self.C = C
        self.R = R
        self.R_start = R_start

        self.model = gp.Model("milp")
        self.x = self.model.addMVar((C, R), lb=-mem_q_lvl + 1, ub=mem_q_lvl - 1,
                                    vtype=GRB.INTEGER, name="x")
        self.cvx_saf0_pos = self.model.addMVar((C, R), lb=0, ub=1,
                                    vtype=GRB.BINARY, name="saf0_pos")
        self.cvx_saf0_neg = self.model.addMVar((C, R), lb=0, ub=1,
                                    vtype=GRB.BINARY, name="saf0_neg")
        self.cvx_q_code = self.model.addMVar((R - R_start,), lb=0, ub=max_q_code,
                                    vtype=GRB.INTEGER, name="q_code")

        # Directly calculate x_corrupt
        self.x_corrupt = self.x + self.cvx_saf0_pos * (
            self.mem_q_lvl - 1
        ) - self.cvx_saf0_neg * (self.mem_q_lvl - 1)
        self.q_code_corrupt = self.rank_vec @ self.x_corrupt @ self.cumsum_mat
        self.q_code_residual = self.q_code_corrupt - self.cvx_q_code

        self.t = self.model.addMVar((R - R_start,), lb=0, vtype=GRB.CONTINUOUS, name="t")

        self.model.setObjective(gp.quicksum(self.t), GRB.MINIMIZE)
        self.model.addConstr(self.t >= 0)
        self.model.addConstr(self.t >= self.q_code_residual)
        self.model.addConstr(self.t >= -self.q_code_residual)

        self.model.Params.Threads = 1
        self.model.Params.OutputFlag = 0
        self.model.Params.LogToConsole = 0
        self.model.Params.LogFile = "gurobi.log"

    def _setup_problem(self, all_fault_pos, all_fault_neg, saf0_pos, saf0_neg, q_code):
        """Sets up the optimization problem constraints."""
        self.cvx_q_code.UB = q_code
        self.cvx_q_code.LB = q_code
        self.cvx_saf0_pos.UB = saf0_pos
        self.cvx_saf0_pos.LB = saf0_pos
        self.cvx_saf0_neg.UB = saf0_neg
        self.cvx_saf0_neg.LB = saf0_neg

        both_no_fault = all_fault_pos & all_fault_neg
        only_pos_fault = all_fault_pos & (~all_fault_neg)
        only_neg_fault = (~all_fault_pos) & all_fault_neg

        self.x.UB = (both_no_fault | only_pos_fault).astype(int) * (
            self.mem_q_lvl - 1
        )
        self.x.LB = -(both_no_fault | only_neg_fault).astype(int) * (
            self.mem_q_lvl - 1
        )

    def cvx_solve(self, all_fault_pos, all_fault_neg, saf0_pos, saf0_neg, q_code):
        """Solves the optimization problem for a single q_code."""
        self.model.reset()
        self._setup_problem(all_fault_pos, all_fault_neg, saf0_pos, saf0_neg, q_code)
        self.model.optimize()

        # Get the values
        rc = self.x.X
        # Set negative values to 0 and return as rc_pos
        rc_pos = np.round(np.maximum(rc, 0))
        rc_neg = np.round(np.maximum(-rc, 0))
        rc_pair = np.stack([rc_pos, rc_neg], axis=0)
        return rc_pair

    def cvx_solve_for_list(self, all_faults_list, saf0_list, q_code_list):
        """Solves the optimization problem for a list of q_codes."""
        rc_pair_list = np.empty_like(all_faults_list[:, 0], dtype=int)
        num_q_codes = len(q_code_list)
        all_faults_list = np.logical_not(all_faults_list)

        all_fault_pos = all_faults_list[:, 0]
        all_fault_neg = all_faults_list[:, 1]
        saf0_pos = saf0_list[:, 0]
        saf0_neg = saf0_list[:, 1]

        for i in range(num_q_codes):
            rc_pair = self.cvx_solve(
                all_fault_pos[i],
                all_fault_neg[i],
                saf0_pos[i],
                saf0_neg[i],
                q_code_list[i],
            )
            if rc_pair[0] is None:
                print(f"Failed to solve for q_code {i}")
                continue
            rc_pair_list[i] = rc_pair

        rc_pos = np.round(np.maximum(rc_pair_list, 0))
        rc_neg = np.round(np.maximum(-rc_pair_list, 0))
        rc_pair_list = np.stack([rc_pos, rc_neg], axis=1)

        return rc_pair_list

# class GurobiFree4:
#     def __init__(self, C, R, mem_q_lvl, R_start, rank_vec, max_q_code, num_processes=1, num_threads=1):
#         self.rank_vec = rank_vec
#         self.cumsum_mat = np.triu(np.ones((R, R)))[:, R_start:]
#         self.mem_q_lvl = mem_q_lvl
#         self.C = C
#         self.R = R
#         self.R_start = R_start
#         self.max_q_code = max_q_code
#         self.num_processes = num_processes
#         self.num_threads = num_threads
#         self.models = [self.create_model() for _ in range(self.num_processes)]

#     def create_model(self):
#         model = gp.Model("milp")
#         x = model.addMVar((self.C, self.R), lb=-self.mem_q_lvl+1, ub=self.mem_q_lvl-1,
#                           vtype=gp.GRB.INTEGER, name="x")
#         cvx_saf0_pos = model.addMVar((self.C, self.R), lb=0, ub=1,
#                                      vtype=gp.GRB.BINARY, name="saf0_pos")
#         cvx_saf0_neg = model.addMVar((self.C, self.R), lb=0, ub=1,
#                                      vtype=gp.GRB.BINARY, name="saf0_neg")
#         cvx_q_code = model.addMVar((self.R - self.R_start,), lb=0, ub=self.max_q_code,
#                                    vtype=gp.GRB.INTEGER, name="q_code")

#         x_pos_corrupt = cvx_saf0_pos * (self.mem_q_lvl - 1)
#         x_neg_corrupt = cvx_saf0_neg * (self.mem_q_lvl - 1)
#         x_corrupt = x + x_pos_corrupt - x_neg_corrupt
#         q_code_corrupt = self.rank_vec @ x_corrupt @ self.cumsum_mat
#         q_code_residual = q_code_corrupt - cvx_q_code

#         t = model.addMVar((self.R - self.R_start,), lb=0, vtype=gp.GRB.CONTINUOUS, name="t")

#         model.setObjective(gp.quicksum(t), gp.GRB.MINIMIZE)
#         model.addConstr(t >= 0)
#         model.addConstr(t >= q_code_residual)
#         model.addConstr(t >= -q_code_residual)

#         model.Params.Threads = self.num_threads
#         model.Params.OutputFlag = 0
#         model.Params.LogToConsole = 0
#         model.Params.LogFile = "gurobi-4.log" 

#         return model, x, cvx_saf0_pos, cvx_saf0_neg, cvx_q_code

#     def _cvx_solve(self, args):
#         # Unpack arguments
#         x_ub, x_lb, saf0_pos, saf0_neg, q_code, proc_id = args
        
#         # Create a new model for this process
#         model, x, cvx_saf0_pos, cvx_saf0_neg, cvx_q_code = self.models[proc_id]


#         model.reset()

#         # Fix the values for cvx_q_code, cvx_saf0_pos, and cvx_saf0_neg
#         cvx_q_code.UB = q_code
#         cvx_q_code.LB = q_code
#         cvx_saf0_pos.UB = saf0_pos
#         cvx_saf0_pos.LB = saf0_pos
#         cvx_saf0_neg.UB = saf0_neg
#         cvx_saf0_neg.LB = saf0_neg

#         # Set bounds for x
#         x.UB = x_ub
#         x.LB = x_lb

#         # Optimize the model
#         model.optimize()

#         # Check if the model solved successfully
#         if model.Status != gp.GRB.OPTIMAL:
#             print(f"Optimization was unsuccessful. Model status: {model.Status}")
#             return None

#         # Get the results
#         rc = x.X
#         return rc

#     def cvx_solve_for_list(self, all_faults_list, saf0_list, q_code_list):
#         num_q_codes = len(q_code_list)
#         all_faults_list = np.logical_not(all_faults_list)

#         all_fault_pos = all_faults_list[:, 0]
#         all_fault_neg = all_faults_list[:, 1]
#         saf0_pos = saf0_list[:, 0]
#         saf0_neg = saf0_list[:, 1]

#         both_no_fault = all_fault_pos & all_fault_neg
#         only_pos_fault = all_fault_pos & (~all_fault_neg)
#         only_neg_fault = (~all_fault_pos) & all_fault_neg

#         x_ub = (both_no_fault | only_pos_fault).astype(int)
#         x_lb = -(both_no_fault | only_neg_fault).astype(int)

#         # Prepare arguments for multiprocessing
#         args = [(x_ub[i], x_lb[i], saf0_pos[i], saf0_neg[i], q_code_list[i], i % self.num_processes) 
#                 for i in range(num_q_codes)]

#         with Pool(processes=self.num_processes) as pool:
#             rc_pair_list = pool.map(self._cvx_solve, args)

#         # Convert to numpy arrays and process results
#         rc_pair_list = np.array([rc for rc in rc_pair_list if rc is not None])

#         if len(rc_pair_list) == 0:
#             return np.array([])  # Return empty array if no valid pairs were found
        
#         rc_pos = np.round(np.maximum(rc_pair_list, 0))
#         rc_neg = np.round(np.maximum(-rc_pair_list, 0))
#         rc_pair_list = np.stack([rc_pos, rc_neg], axis=1)

#         return rc_pair_list
