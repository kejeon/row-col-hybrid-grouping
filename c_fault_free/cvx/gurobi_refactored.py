import numpy as np
import gurobipy as gp
from gurobipy import GRB
import multiprocessing as mp
from functools import partial
from typing import Type, Tuple, List, Any
from line_profiler import LineProfiler


# Initialize a Gurobi environment with custom settings
env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.setParam('LogToConsole', 0)
env.setParam('LogFile', 'gurobi_sparsify_log_file.log')
env.setParam('NodeLimit', 5)
env.setParam('TimeLimit', 1)
env.setParam('Threads', 1)
env.setParam('UpdateMode', 1)
env.setParam('Presolve', 1)
env.start()

def flatten_list(l):
    if not isinstance(l, list):
        return [l]
    if not isinstance(l[0], list):
        return l
    return [var for sublist in l for var in sublist]


class BaseGurobiOptimizer:
    def __init__(self, C: int, R: int, mem_q_lvl: int, R_start: int, pos_neg_sep: bool,
                 rank_vec: np.ndarray, max_q_code: int, debug: bool = False) -> None:
        self.C: int = C
        self.R: int = R
        self.mem_q_lvl: int = mem_q_lvl
        self.R_start: int = R_start
        self.rank_vec: np.ndarray = rank_vec
        self.max_q_code: int = max_q_code
        self.cumsum_mat: np.ndarray = np.triu(np.ones((R, R)))[:, R_start:]
        self.pos_neg_sep: bool = pos_neg_sep

        # Setup logging
        self.debug = debug
        if self.debug:
            env.setParam('OutputFlag', 1)

        # Initialize the model
        self.model: gp.Model = gp.Model("milp", env=env)
        self._initialize_variables()

        if self.pos_neg_sep:
            self.setup_problem = self.setup_problem_sep
            self.get_solution = self.get_solution_sep
        else:
            self.setup_problem = self.setup_problem_no_sep
            self.get_solution = self.get_solution_no_sep


    def _initialize_variables(self) -> None:
        if self.pos_neg_sep:
            self.x_pos: gp.MVar = self.model.addMVar(
                (self.C, self.R), lb=0, ub=self.mem_q_lvl - 1, 
                vtype=GRB.INTEGER, name="x_pos")
            self.x_neg: gp.MVar = self.model.addMVar(
                (self.C, self.R), lb=0, ub=self.mem_q_lvl - 1,
                vtype=GRB.INTEGER, name="x_neg")
            self.x = self.x_pos - self.x_neg
            self.flat_x_pos_list = flatten_list(self.x_pos.tolist())
            self.flat_x_neg_list = flatten_list(self.x_neg.tolist())
            self.flat_x_list = self.flat_x_pos_list + self.flat_x_neg_list
        else:
            self.x: gp.MVar = self.model.addMVar(
                (self.C, self.R), lb=-(self.mem_q_lvl - 1), ub=self.mem_q_lvl - 1,
                vtype=GRB.INTEGER, name="x")
            self.flat_x_list = flatten_list(self.x.tolist())
            
        self.q_code: gp.MVar = self.model.addMVar(
            (self.R - self.R_start,), lb=0, ub=self.max_q_code,
            vtype=GRB.INTEGER, name="q_code")
        self.flat_q_code_list = flatten_list(self.q_code.tolist())


    def _define_problem(self) -> None:
        raise NotImplementedError
    

    def setup_problem_sep(self, not_saf1_pos: np.ndarray, not_saf1_neg: np.ndarray,
                          saf0_pos: np.ndarray, saf0_neg: np.ndarray, q_code: np.ndarray) -> None:
        x_pos_ub = (not_saf1_pos.astype(int) * (self.mem_q_lvl - 1)).flatten().tolist()
        x_neg_ub = (not_saf1_neg.astype(int) * (self.mem_q_lvl - 1)).flatten().tolist()
        x_pos_lb = (saf0_pos.astype(int) * (self.mem_q_lvl - 1)).flatten().tolist()
        x_neg_lb = (saf0_neg.astype(int) * (self.mem_q_lvl - 1)).flatten().tolist()
        
        self.model.setAttr(GRB.Attr.UB, 
                           self.flat_q_code_list + self.flat_x_list, 
                           q_code.flatten().tolist() + x_pos_ub + x_neg_ub)
        self.model.setAttr(GRB.Attr.LB, 
                           self.flat_q_code_list + self.flat_x_list, 
                           q_code.flatten().tolist() + x_pos_lb + x_neg_lb)


    def setup_problem_no_sep(self, not_saf1_pos: np.ndarray, not_saf1_neg: np.ndarray,
                             saf0_pos: np.ndarray, saf0_neg: np.ndarray, q_code: np.ndarray) -> None:
        saf1_pos = np.logical_not(saf0_pos)
        saf1_neg = np.logical_not(saf0_neg)
        no_fault_pos = np.logical_not(np.logical_or(saf0_pos, saf1_pos))
        no_fault_neg = np.logical_not(np.logical_or(saf0_neg, saf1_neg))

        x_ub = (
            -(self.mem_q_lvl - 1) * (saf1_pos * saf0_neg) +
            (self.mem_q_lvl - 1) * ((saf1_neg + no_fault_neg) * (saf0_pos + no_fault_pos))
            ).flatten().tolist()
        x_lb = (
            -(self.mem_q_lvl - 1) * ((saf1_pos + no_fault_pos) * (saf0_neg + no_fault_neg)) +
            (self.mem_q_lvl - 1) * (saf1_neg * saf0_pos)
            ).flatten().tolist()
        
        self.model.setAttr(GRB.Attr.UB, 
                           self.flat_q_code_list + self.flat_x_list, 
                           q_code.flatten().tolist() + x_ub)
        self.model.setAttr(GRB.Attr.LB, 
                           self.flat_q_code_list + self.flat_x_list, 
                           q_code.flatten().tolist() + x_lb)


    def optimize(self) -> None:
        try:
            self.model.optimize()
        except gp.GurobiError as e:
            raise ValueError(f"Gurobi error: {e}")
        

    def solve_single(self, fault_pos, fault_neg, saf0_pos, saf0_neg, q_code):
        self.model.reset()
        self.setup_problem(fault_pos, fault_neg, saf0_pos, saf0_neg, q_code)
        self.optimize()
        return self.get_solution()
    

    def solve_multiple(self, faults_list, saf0_list, q_code_list):
        num_q_codes = len(q_code_list)
        results = np.empty_like(faults_list, dtype=int)
        inverted_faults = np.logical_not(faults_list)
        # inverted_faults = faults_list

        for i in range(num_q_codes):
            result = self.solve_single(
                inverted_faults[i][0],
                inverted_faults[i][1],
                saf0_list[i][0],
                saf0_list[i][1],
                q_code_list[i],
            )
            results[i] = result
        return results


    def get_solution_sep(self) -> np.ndarray:
        if self.model.SolCount == 0:
            print("WARNING: No solution found")
            return np.zeros((2, self.C, self.R))
        solutions = self.model.getAttr(GRB.Attr.X, self.flat_x_list)
        rc_sol = np.array(solutions).reshape(2, self.C, self.R)
        return rc_sol
    
    
    def get_solution_no_sep(self) -> np.ndarray:
        if self.model.SolCount == 0:
            print("WARNING: No solution found")
            return np.zeros((2, self.C, self.R))
        # solutions = self.model.getAttr(GRB.Attr.X, self.flat_x_list)
        # rc_sol = np.array(solutions).reshape(self.C, self.R)
        rc: np.ndarray = self.x.X
        rc_pos: np.ndarray = np.round(np.maximum(rc, 0))
        rc_neg: np.ndarray = np.round(np.maximum(-rc, 0))
        return np.stack([rc_pos, rc_neg], axis=0)

    
class gurobi_fawd(BaseGurobiOptimizer):
    def __init__(self, C: int, R: int, mem_q_lvl: int, R_start: int, rank_vec: np.ndarray, max_q_code: int) -> None:
        super().__init__(C, R, mem_q_lvl, R_start, True, rank_vec, max_q_code)
        self._define_problem()
    
    def _define_problem(self) -> None:
        q_code_corrupt = (self.rank_vec @ self.x).sum()
        q_code_residual = self.q_code - q_code_corrupt

        self.model.setObjective(self.x_pos.sum() + self.x_neg.sum(), GRB.MINIMIZE)
        self.model.addConstr(q_code_residual == 0)

class gurobi_cvm_relaxed(BaseGurobiOptimizer):
    def __init__(self, C: int, R: int, mem_q_lvl: int, R_start: int, rank_vec: np.ndarray, max_q_code: int) -> None:
        super().__init__(C, R, mem_q_lvl, R_start, True, rank_vec, max_q_code)
        self._define_problem()
    
    def _define_problem(self) -> None:
        q_code_corrupt = (self.rank_vec @ self.x).sum()
        q_code_residual = self.q_code - q_code_corrupt
        t = self.model.addVar(lb=0, vtype=GRB.INTEGER, name="t")

        self.model.setObjective(t, GRB.MINIMIZE)
        self.model.addConstr(t >= q_code_residual)
        self.model.addConstr(t >= -q_code_residual)

class gurobi_cvm(BaseGurobiOptimizer):
    def __init__(self, C: int, R: int, mem_q_lvl: int, R_start: int, rank_vec: np.ndarray, max_q_code: int) -> None:
        super().__init__(C, R, mem_q_lvl, R_start, True, rank_vec, max_q_code)
        self._define_problem()
    
    def _define_problem(self) -> None:
        q_code_corrupt = (self.rank_vec @ self.x).sum()
        q_code_residual = self.q_code - q_code_corrupt
        t = self.model.addVar(lb=0, vtype=GRB.INTEGER, name="t")
        lambda_s = 1/(2*self.R*self.C*(self.mem_q_lvl -1))
        sparsity = lambda_s*(self.x_pos.sum() + self.x_neg.sum())

        self.model.setObjective(t + sparsity, GRB.MINIMIZE)
        self.model.addConstr(t >= q_code_residual)
        self.model.addConstr(t >= -q_code_residual)

class gurobi_cvm_int(BaseGurobiOptimizer):
    def __init__(self, C: int, R: int, mem_q_lvl: int, R_start: int, rank_vec: np.ndarray, max_q_code: int) -> None:
        super().__init__(C, R, mem_q_lvl, R_start, False, rank_vec, max_q_code)
        self._define_problem()
    
    def _define_problem(self) -> None:
        q_code_corrupt = (self.rank_vec @ self.x).sum()
        q_code_residual = self.q_code - q_code_corrupt
        t = self.model.addVar(lb=0, vtype=GRB.INTEGER, name="t")

        self.model.setObjective(t, GRB.MINIMIZE)
        self.model.addConstr(t >= q_code_residual)
        self.model.addConstr(t >= -q_code_residual)

def gurobi_solve_multiple(
    all_faults: np.ndarray,
    saf0: np.ndarray,
    q_code: np.ndarray,
    C: int,
    R: int,
    mem_q_lvl: int,
    R_start: int,
    rank_vec: np.ndarray,
    max_q_code: int,
    gurobi_class: Type[Any]
) -> np.ndarray:
    cvx_problem = gurobi_class(C, R, mem_q_lvl, R_start, rank_vec, max_q_code)
    # profiler = LineProfiler()
    # profiler.add_function(cvx_problem.setup_problem)
    # profiler.add_function(cvx_problem.solve_single)
    # profiler.add_function(cvx_problem.get_solution)
    # profiler_wrapper = profiler(cvx_problem.solve_multiple)
    # cvx_rc = profiler_wrapper(all_faults, saf0, q_code)
    # profiler.print_stats()
    cvx_rc = cvx_problem.solve_multiple(all_faults, saf0, q_code)

    return cvx_rc

def gurobi_solve_multiple_parallel(
    unmatched_all_faults: np.ndarray,
    unmatched_saf0: np.ndarray,
    unmatched_test_q_codes: np.ndarray,
    C: int,
    R: int,
    mem_q_lvl: int,
    R_start: int,
    rank_vec: np.ndarray,
    max_q_code: int,
    gurobi_class: Type[Any],
    num_pools: int = 20
) -> np.ndarray:
    # Prepare arguments for each process
    num_batches = num_pools * 1
    num_examples = len(unmatched_test_q_codes)
    num_examples_per_pool = num_examples // num_batches

    start_idx = [i * num_examples_per_pool for i in range(num_batches)]
    end_idx = [(i + 1) * num_examples_per_pool for i in range(num_batches)]
    end_idx[-1] = num_examples

    args_list = [
        (
            unmatched_all_faults[start_idx[i]:end_idx[i]],
            unmatched_saf0[start_idx[i]:end_idx[i]],
            unmatched_test_q_codes[start_idx[i]:end_idx[i]]
        )
        for i in range(num_batches)
    ]

    gurobi_partial = partial(
        gurobi_solve_multiple,
        gurobi_class=gurobi_class,
        C=C,
        R=R,
        mem_q_lvl=mem_q_lvl,
        R_start=R_start,
        rank_vec=rank_vec,
        max_q_code=max_q_code
    )

    with mp.Pool(num_pools) as pool:    
        results = pool.starmap(gurobi_partial, args_list)

    cvx_rc = np.concatenate(results, axis=0)

    return cvx_rc