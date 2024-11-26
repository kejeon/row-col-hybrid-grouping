import numpy as np
import gurobipy as gp
from gurobipy import GRB
import multiprocessing as mp
from functools import partial
from typing import Type, Tuple, List, Any

from rc_grouping.rccodes import RcCodes, gen_conversion_vector

import logging
from line_profiler import LineProfiler

from itertools import chain

# Initialize a Gurobi environment with custom settings
env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.setParam('LogToConsole', 0)
env.setParam('LogFile', 'gurobi_sparsify_log_file.log')
env.setParam('NodeLimit', 5)
# env.setParam('IterationLimit', 1)
env.setParam('TimeLimit', 1)
env.setParam('Threads', 1)
# env.setParam('Cuts', 0)
env.setParam('UpdateMode', 1)
env.setParam('Presolve', 1)
# env.setParam('MIPFocus', 1)
env.start()


class BaseGurobiOptimizer:
    """
    Base class for Gurobi-based optimization models.

    This class provides a foundation for setting up and solving mixed-integer
    linear programming (MILP) problems using Gurobi. It handles the basic model
    setup, variable initialization, and common optimization tasks.

    Attributes:
        C (int): Number of columns in the decision variable matrix.
        R (int): Number of rows in the decision variable matrix.
        mem_q_lvl (int): Memory quantization level, defining the bounds for decision variables.
        R_start (int): Index indicating the starting row for cumulative sums in calculations.
        rank_vec (np.ndarray): Rank vector used in calculating the corrupted q_code.
        max_q_code (int): Maximum allowable value for q_code variables.
        cumsum_mat (np.ndarray): Matrix used for cumulative sum calculations.
        model (gp.Model): The Gurobi optimization model.
        debug (bool): Flag to enable or disable debug mode.
    """
    def __init__(self, C: int, R: int, mem_q_lvl: int, R_start: int, rank_vec: np.ndarray, max_q_code: int, debug: bool = False) -> None:
        """
        Initializes the base optimizer with model parameters.

        Args:
            C (int): Number of columns in the decision variable matrix.
            R (int): Number of rows in the decision variable matrix.
            mem_q_lvl (int): Memory quantization level, defining the bounds for decision variables.
            R_start (int): Index indicating the starting row for cumulative sums in calculations.
            rank_vec (np.ndarray): Rank vector used in calculating the corrupted q_code.
            max_q_code (int): Maximum allowable value for q_code variables.
            debug (bool): Flag to enable or disable debug mode. Defaults to False.
        """
        self.C: int = C
        self.R: int = R
        self.mem_q_lvl: int = mem_q_lvl
        self.R_start: int = R_start
        self.rank_vec: np.ndarray = rank_vec
        self.max_q_code: int = max_q_code
        self.cumsum_mat: np.ndarray = np.triu(np.ones((R, R)))[:, R_start:]

        # Initialize the model
        self.model: gp.Model = gp.Model("milp", env=env)
        self._initialize_variables()

        # Setup logging
        self.debug = debug
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

    def _initialize_variables(self) -> None:
        """
        Initialize the decision variables for the model.

        This method sets up the decision variables used in the optimization
        model, including the main decision variable matrix `x`, safety flags
        `saf0_pos` and `saf0_neg`, and the q_code variable.
        """
        # self.x: gp.MVar = self.model.addMVar((self.C, self.R), lb=-self.mem_q_lvl + 1, ub=self.mem_q_lvl - 1,
                                            #  vtype=GRB.INTEGER, name="x")
        self.x_pos: gp.MVar = self.model.addMVar((self.C, self.R), lb=0, ub=self.mem_q_lvl - 1,
                                             vtype=GRB.INTEGER, name="x_pos")
        self.x_neg: gp.MVar = self.model.addMVar((self.C, self.R), lb=0, ub=self.mem_q_lvl - 1,
                                             vtype=GRB.INTEGER, name="x_neg")
        self.saf0_pos: gp.MVar = self.model.addMVar((self.C, self.R), lb=0, ub=1,
                                                    vtype=GRB.BINARY, name="saf0_pos")
        self.saf0_neg: gp.MVar = self.model.addMVar((self.C, self.R), lb=0, ub=1,
                                                    vtype=GRB.BINARY, name="saf0_neg")
        self.q_code: gp.MVar = self.model.addMVar((self.R - self.R_start,), lb=0, ub=self.max_q_code,
                                                  vtype=GRB.INTEGER, name="q_code")
        self.x = self.x_pos - self.x_neg

    def _setup_constraints(self) -> None:
        """
        Setup constraints for the optimization problem.

        This method adds constraints to the model, which include ensuring the
        non-negativity of the residuals and bounding the decision variables.
        """
        # Calculate corrupted x values and residuals
        x_corrupt: gp.MVar = self.x + self.saf0_pos * (self.mem_q_lvl - 1) - self.saf0_neg * (self.mem_q_lvl - 1)
        q_code_corrupt: gp.MVar = (self.rank_vec @ x_corrupt).sum()
        q_code_residual: gp.MVar = q_code_corrupt - self.q_code
        self.model.setObjective(self.x_pos.sum() + self.x_neg.sum(), GRB.MINIMIZE)
        self.model.addConstr(q_code_residual == 0)


    def optimize(self) -> None:
        try:
            self.model.optimize()
        except gp.GurobiError as e:
            logging.error(f"Optimization failed: {e}")
            raise


    def get_solution(self) -> np.ndarray:
        # rc_pos = self.x_pos.X
        # rc_neg = self.x_neg.X
        vars_list_x_pos = self.x_pos.tolist()
        flat_vars_list_x_pos = [var for sublist in vars_list_x_pos for var in sublist]
        vars_list_x_neg = self.x_neg.tolist()
        flat_vars_list_x_neg = [var for sublist in vars_list_x_neg for var in sublist]
        solutions = self.model.getAttr(GRB.Attr.X, flat_vars_list_x_pos + flat_vars_list_x_neg)
        rc_sol = np.array(solutions).reshape(2, self.C, self.R)
        # rc_pos = np.array(solutions[:len(flat_vars_list_x_pos)]).reshape(self.C, self.R)
        # rc_neg = np.array(solutions[len(flat_vars_list_x_pos):]).reshape(self.C, self.R)
        return rc_sol


class GurobiOptimizer(BaseGurobiOptimizer):
    """
    GurobiOptimizer is a derived class for solving specific optimization problems.

    This class builds on BaseGurobiOptimizer by implementing problem-specific
    setup and solution methods, tailored to a particular use case.
    """
    def __init__(self, C: int, R: int, mem_q_lvl: int, R_start: int, rank_vec: np.ndarray, max_q_code: int, debug: bool = False) -> None:
        """
        Initializes the GurobiOptimizer with specific model parameters.
        """
        super().__init__(C, R, mem_q_lvl, R_start, rank_vec, max_q_code, debug)
        self._setup_constraints()

    def setup_problem(self, fault_pos: np.ndarray, fault_neg: np.ndarray,
                      saf0_pos: np.ndarray, saf0_neg: np.ndarray, q_code: np.ndarray) -> None:
        """
        Configures the problem with given fault and saf0 constraints.

        This method sets the bounds and constraints for the decision variables
        based on the provided all fault and saf0 positions, and q_code values.

        Args:
            fault_pos (np.ndarray): Binary array indicating positions of all faults on positive array.
            fault_neg (np.ndarray): Binary array indicating positions of all faults on negative array.
            saf0_pos (np.ndarray): Binary array indicating positions of SAF0 on positive array.
            saf0_neg (np.ndarray): Binary array indicating positions of SAF0 on negative array.
            q_code (np.ndarray): Target q_code value to constrain the optimization.
        """
        flat_vars_list_q_code = self.q_code.tolist()
        vars_list_saf0_pos = self.saf0_pos.tolist()
        flat_vars_list_saf0_pos = [var for sublist in vars_list_saf0_pos for var in sublist]
        vars_list_saf0_neg = self.saf0_neg.tolist()
        flat_vars_list_saf0_neg = [var for sublist in vars_list_saf0_neg for var in sublist]
        vars_list_x_pos = self.x_pos.tolist()
        flat_vars_list_x_pos = [var for sublist in vars_list_x_pos for var in sublist]
        vars_list_x_neg = self.x_neg.tolist()
        flat_vars_list_x_neg = [var for sublist in vars_list_x_neg for var in sublist]

        all_vars_list = flat_vars_list_q_code + flat_vars_list_saf0_pos + flat_vars_list_saf0_neg
        all_values_list = q_code.flatten().tolist() + saf0_pos.flatten().tolist() + saf0_neg.flatten().tolist()
        x_pos_values = (fault_pos.astype(int) * (self.mem_q_lvl - 1)).flatten().tolist()
        x_neg_values = (fault_neg.astype(int) * (self.mem_q_lvl - 1)).flatten().tolist()
        
        self.model.setAttr(GRB.Attr.UB, 
                           all_vars_list + flat_vars_list_x_pos + flat_vars_list_x_neg, 
                           all_values_list + x_pos_values + x_neg_values)
        self.model.setAttr(GRB.Attr.LB, 
                           all_vars_list, 
                           all_values_list)

        # self.model.update()

    def solve_single(self, fault_pos: np.ndarray, fault_neg: np.ndarray,
                     saf0_pos: np.ndarray, saf0_neg: np.ndarray, q_code: np.ndarray) -> np.ndarray:
        """
        Solves the optimization problem for a single set of inputs.

        This method optimizes the model for a specific configuration of faults,
        saf0 flags, and q_code values, and returns the solution.

        Args:
            fault_pos (np.ndarray): Binary array indicating positions with positive faults.
            fault_neg (np.ndarray): Binary array indicating positions with negative faults.
            saf0_pos (np.ndarray): Binary array indicating saf0 flags for positive values.
            saf0_neg (np.ndarray): Binary array indicating saf0 flags for negative values.
            q_code (np.ndarray): Array of q_code values to constrain the optimization.

        Returns:
            np.ndarray: A 2D array containing the positive and negative parts
                        of the decision variable matrix.
        """
        self.model.reset()
        self.setup_problem(fault_pos, fault_neg, saf0_pos, saf0_neg, q_code)
        self.optimize()
        return self.get_solution()

    def solve_multiple(
            self, 
            faults_list: np.ndarray, 
            saf0_list: np.ndarray, 
            q_code_list: np.ndarray) -> np.ndarray:
        """
        Solves the optimization problem for multiple sets of inputs using a simple for loop.

        This method runs the optimization for a list of configurations sequentially,
        each defined by its own set of faults, saf0 flags, and q_code values.

        Args:
            faults_list (np.ndarray): Array of fault configurations (shape: [n, 2, C, R]).
            saf0_list (np.ndarray): Array of saf0 configurations (shape: [n, 2, C, R]).
            q_code_list (np.ndarray): Array of q_code values (shape: [n, R-R_start]).

        Returns:
            np.ndarray: A 3D array containing the positive and negative parts
                        of the decision variable matrices for each configuration.
        """
        num_q_codes: int = len(q_code_list)
        results: np.ndarray = np.empty_like(faults_list, dtype=int)

        # Invert faults for easier computation
        inverted_faults: np.ndarray = np.logical_not(faults_list)

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
    
    # solve_multiple = base_solve_multiple
    # solve_multiple = solve_multiple_with_forloop
    # solve_multiple = solve_multiple_with_list_comp


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
    gurobi_class: Type[Any] = GurobiOptimizer
) -> np.ndarray:
    """
    Helper function to initialize and run the Gurobi optimization for multiple inputs.

    Args:
        gurobi_class: The class implementing the Gurobi optimization.
        all_faults, saf0, q_code: Numpy arrays containing inputs for the Gurobi optimization.
        C, R, mem_q_lvl, R_start: Integer parameters related to the optimization.
        rank_vec: Numpy array used in the Gurobi optimization.
        max_q_code: Integer indicating the maximum q_code value.

    Returns:
        np.ndarray: The result of the solve_multiple method of the provided Gurobi class.
    """
    cvx_problem = gurobi_class(C, R, mem_q_lvl, R_start, rank_vec, max_q_code)
    
    profiler = LineProfiler()
    profiler.add_function(GurobiOptimizer.setup_problem)
    profiler.add_function(GurobiOptimizer.solve_single)
    profiler.add_function(GurobiOptimizer.get_solution)
    profiler_wrapper = profiler(cvx_problem.solve_multiple)
    cvx_rc = profiler_wrapper(all_faults, saf0, q_code)
    profiler.print_stats()
    
    # cvx_rc = cvx_problem.solve_multiple(all_faults, saf0, q_code)

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
    gurobi_class: Type[Any] = GurobiOptimizer,
    num_pools: int = 1
) -> np.ndarray:
    """
    Runs the Gurobi optimization in parallel using multiprocessing.

    Args:
        faultmaps: An object containing the fault maps.
        unmatched: List or array of indices that specify unmatched cases.
        test_q_codes: Array of test q_codes.
        codebook: Codebook object containing q_code mappings.
        C: Integer representing the number of columns in the decision variable matrix.
        R: Integer representing the number of rows in the decision variable matrix.
        mem_q_lvl: Memory quantization level.
        R_start: Starting row index for cumulative sums.
        shift_base: Base used for generating the conversion vector.
        gurobi_class: The class implementing the Gurobi optimization (should have a solve_multiple method).
        num_pools: The number of parallel processes to run (default is 20).

    Returns:
        Numpy array containing the results of the Gurobi optimization for all inputs.
    """
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

    with mp.get_context("fork").Pool(num_pools) as pool:    
        results = pool.starmap(gurobi_partial, args_list)

    cvx_rc = np.concatenate(results, axis=0)

    return cvx_rc

