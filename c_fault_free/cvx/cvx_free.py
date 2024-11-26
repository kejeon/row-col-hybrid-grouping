import numpy as np
import cvxpy as cp

class cvx_free_base():
    def __init__(self, C, R, mem_q_lvl, R_start, rank_vec, max_q_code, solver):
        self.solver = getattr(cp, solver)
        self.x_pos = cp.Variable(shape=(C,R))
        self.x_neg = cp.Variable(shape=(C,R))

        self.cvx_all_fault_pos = cp.Parameter(shape=(C, R), nonneg=True)
        self.cvx_all_fault_neg = cp.Parameter(shape=(C, R), nonneg=True)
        self.cvx_saf0_pos = cp.Parameter(shape=(C, R), nonneg=True)
        self.cvx_saf0_neg = cp.Parameter(shape=(C, R), nonneg=True)
        self.cvx_q_code = cp.Parameter(shape=(R,))

        self.x_pos_corrupt = cp.multiply(self.x_pos, self.cvx_all_fault_pos) + self.cvx_saf0_pos*(mem_q_lvl)
        self.x_neg_corrupt = cp.multiply(self.x_neg, self.cvx_all_fault_neg) + self.cvx_saf0_neg*(mem_q_lvl)

        self.x_corrupt = self.x_pos_corrupt - self.x_neg_corrupt
        self.q_code_corrupt = rank_vec @ self.x_corrupt @ np.triu(np.ones((R,R)))[:,R_start:]
        self.q_code_residual = cp.multiply(self.q_code_corrupt - self.cvx_q_code, 1/max_q_code)

    def cvx_solve(self, all_fault_pos, all_fault_neg, saf0_pos, saf0_neg, q_code):
        return
    
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




class cvx_free_1(cvx_free_base):
    def __init__(self, C, R, mem_q_lvl, R_start, rank_vec, max_q_code, solver):
        super().__init__(C, R, mem_q_lvl, R_start, rank_vec, max_q_code, solver)
        self.least_sq = cp.norm(self.q_code_residual, 2)
        self.l1_reg = (cp.norm(self.x_pos, 1) + cp.norm(self.x_neg, 1)) / (C*R*2*(mem_q_lvl-1))
        self.l1_reg_lambda = 1e-7
        self.obj = cp.Minimize(self.least_sq + self.l1_reg_lambda*self.l1_reg)
        self.constraints = [self.x_pos >= 0, 
                            self.x_neg >= 0, 
                            self.x_pos <= mem_q_lvl-1, 
                            self.x_neg <= mem_q_lvl-1]
        self.prob = cp.Problem(self.obj, self.constraints)


    def cvx_solve(self, all_fault_pos, all_fault_neg, saf0_pos, saf0_neg, q_code):
        self.cvx_q_code.value = q_code
        self.cvx_all_fault_pos.value = np.logical_not(all_fault_pos)
        self.cvx_all_fault_neg.value = np.logical_not(all_fault_neg)
        self.cvx_saf0_pos.value = saf0_pos
        self.cvx_saf0_neg.value = saf0_neg

        self.prob.solve(solver=self.solver)

        rc_pos = self.x_pos.value
        rc_neg = self.x_neg.value
        rc_pair = np.stack([rc_pos, rc_neg], axis=0)
        return rc_pair
    

class cvx_free_2(cvx_free_base):
    def __init__(self, C, R, mem_q_lvl, R_start, rank_vec, max_q_code, solver):
        super().__init__(C, R, mem_q_lvl, R_start, rank_vec, max_q_code, solver)
        self.x_pos = cp.Variable(shape=(C,R), boolean=True)
        self.x_neg = cp.Variable(shape=(C,R), boolean=True)

        self.cvx_all_fault_pos = cp.Parameter(shape=(C, R), nonneg=True)
        self.cvx_all_fault_neg = cp.Parameter(shape=(C, R), nonneg=True)
        self.cvx_saf0_pos = cp.Parameter(shape=(C, R), nonneg=True)
        self.cvx_saf0_neg = cp.Parameter(shape=(C, R), nonneg=True)
        self.cvx_q_code = cp.Parameter(shape=(R-R_start,))

        self.x_pos_corrupt = cp.multiply(self.x_pos, self.cvx_all_fault_pos) + self.cvx_saf0_pos*(mem_q_lvl)
        self.x_neg_corrupt = cp.multiply(self.x_neg, self.cvx_all_fault_neg) + self.cvx_saf0_neg*(mem_q_lvl)

        self.x_corrupt = self.x_pos_corrupt - self.x_neg_corrupt
        self.q_code_corrupt = rank_vec @ self.x_corrupt @ np.triu(np.ones((R,R)))[:,R_start:]
        self.q_code_residual = self.q_code_corrupt - self.cvx_q_code
        # self.q_code_residual = cp.multiply(self.q_code_corrupt - self.cvx_q_code, 1/max_q_code)

        self.least_sq = cp.norm(self.q_code_residual, 1)
        self.l1_reg = (cp.norm(self.x_pos, 1) + cp.norm(self.x_neg, 1)) / (C*R*2*(mem_q_lvl-1))
        # self.l1_reg_lambda = 1e-7
        self.obj = cp.Minimize(self.least_sq)
        self.constraints = [self.l1_reg <= 100]
        self.prob = cp.Problem(self.obj, self.constraints)


    def cvx_solve(self, all_fault_pos, all_fault_neg, saf0_pos, saf0_neg, q_code):
        self.cvx_q_code.value = q_code
        self.cvx_all_fault_pos.value = np.logical_not(all_fault_pos)
        self.cvx_all_fault_neg.value = np.logical_not(all_fault_neg)
        self.cvx_saf0_pos.value = saf0_pos
        self.cvx_saf0_neg.value = saf0_neg

        if self.solver == cp.MOSEK:
            # print("Using MOSEK")
            self.prob.solve(solver=self.solver, 
                            mosek_params={'MSK_IPAR_NUM_THREADS': 1,
                                          'MSK_DPAR_MIO_TOL_ABS_GAP': 1.0e5,
                                          'MSK_DPAR_MIO_TOL_REL_GAP': 1.0e-1,
                                          'MSK_DPAR_MIO_REL_GAP_CONST': 1.0e-1,
                                          'MSK_DPAR_MIO_TOL_ABS_RELAX_INT': 1.0e-1,
                                          'MSK_IPAR_MIO_MAX_NUM_BRANCHES': 3,
                                          'MSK_IPAR_MIO_MAX_NUM_SOLUTIONS': 1,
                                          'MSK_IPAR_PRESOLVE_USE': 0,
                                          'MSK_IPAR_MIO_HEURISTIC_LEVEL': 4},
                            verbose=False)
        else:
            self.prob.solve(solver=self.solver, verbose=False)

        rc_pos = self.x_pos.value
        rc_neg = self.x_neg.value
        rc_pair = np.stack([rc_pos, rc_neg], axis=0)
        return rc_pair


class cvx_free_3(cvx_free_base):
    def __init__(self, C, R, mem_q_lvl, R_start, rank_vec, max_q_code, solver):
        super().__init__(C, R, mem_q_lvl, R_start, rank_vec, max_q_code, solver)
        self.x_pos = cp.Variable(shape=(C,R), integer=True)
        self.x_neg = cp.Variable(shape=(C,R), integer=True)

        self.cvx_all_fault_pos = cp.Parameter(shape=(C, R), nonneg=True)
        self.cvx_all_fault_neg = cp.Parameter(shape=(C, R), nonneg=True)
        self.cvx_saf0_pos = cp.Parameter(shape=(C, R), nonneg=True)
        self.cvx_saf0_neg = cp.Parameter(shape=(C, R), nonneg=True)
        self.cvx_q_code = cp.Parameter(shape=(R-R_start,))

        self.x_pos_corrupt = cp.multiply(self.x_pos, self.cvx_all_fault_pos) + self.cvx_saf0_pos*(mem_q_lvl)
        self.x_neg_corrupt = cp.multiply(self.x_neg, self.cvx_all_fault_neg) + self.cvx_saf0_neg*(mem_q_lvl)

        self.x_corrupt = self.x_pos_corrupt - self.x_neg_corrupt
        self.q_code_corrupt = rank_vec @ self.x_corrupt @ np.triu(np.ones((R,R)))[:,R_start:]
        self.q_code_residual = self.q_code_corrupt - self.cvx_q_code
        # self.q_code_residual = cp.multiply(self.q_code_corrupt - self.cvx_q_code, 1/max_q_code)

        self.least_sq = cp.norm(self.q_code_residual, 1)
        self.l1_reg = (cp.norm(self.x_pos, 1) + cp.norm(self.x_neg, 1)) / (C*R*2*(mem_q_lvl-1))
        # self.l1_reg_lambda = 1e-7
        self.obj = cp.Minimize(self.least_sq)
        self.constraints = [self.x_pos <= mem_q_lvl-1,
                            self.x_neg <= mem_q_lvl-1,
                            self.x_pos >= 0,
                            self.x_neg >= 0]
        self.prob = cp.Problem(self.obj, self.constraints)


    def cvx_solve(self, all_fault_pos, all_fault_neg, saf0_pos, saf0_neg, q_code):
        self.cvx_q_code.value = q_code
        self.cvx_all_fault_pos.value = np.logical_not(all_fault_pos)
        self.cvx_all_fault_neg.value = np.logical_not(all_fault_neg)
        self.cvx_saf0_pos.value = saf0_pos
        self.cvx_saf0_neg.value = saf0_neg

        self.prob.solve(solver=self.solver, verbose=False)

        rc_pos = self.x_pos.value
        rc_neg = self.x_neg.value
        rc_pair = np.stack([rc_pos, rc_neg], axis=0)
        return rc_pair