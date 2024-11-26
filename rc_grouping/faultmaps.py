#%%
import numpy as np
from numba import jit, njit

@njit(cache=True)
def numba_apply_fault(rc_code, map_all_faults, map_saf0, q_lvl):
    # if rc_code.shape[1:] != map_all_faults.shape[1:]:
    #         raise ValueError("Code shape does not match fault map shape")
    # my_map_all_faults = self.map_all_faults[self.idx].reshape((1, ))
    # self.map_saf0, self.map_saf1, self.map_all_faults = self.gen_fault_map()
    # faulty_code = rc_code.copy()
    faulty_code = rc_code * np.logical_not(map_all_faults)
    faulty_code = faulty_code + map_saf0*(q_lvl-1)
    return faulty_code

class FaultMaps():
    def __init__(self, R, C, q_lvl, n=1, p_saf0=0.0175, p_saf1=0.0904, pos_neg_sep=True):
        super().__init__()
        self.R = R
        self.C = C
        self.q_lvl = q_lvl
        self.p_saf0 = p_saf0
        self.p_saf1 = p_saf1
        self.pos_neg_sep = pos_neg_sep
        self.p_no_saf = 1 - p_saf0 - p_saf1

        self.map_saf0_list, self.map_saf1_list, self.map_all_faults_list = self.gen_fault_map(n)
        self.idx = 0
        self.map_saf0, self.map_saf1, self.map_all_faults = self.map_saf0_list[self.idx], self.map_saf1_list[self.idx], self.map_all_faults_list[self.idx]
        self.idx = -1

    def next(self):
        self.idx += 1
        self.idx = self.idx % len(self.map_saf0_list)
        self.map_saf0, self.map_saf1, self.map_all_faults = self.map_saf0_list[self.idx], self.map_saf1_list[self.idx], self.map_all_faults_list[self.idx]
        return 

    def gen_fault_map(self, n=1):
        rand_map = np.random.rand(n, self.C, self.R)
        if self.pos_neg_sep:
            rand_map = np.random.rand(n, 2, self.C, self.R)
        else:
            rand_map = np.random.rand(n, 1, self.C, self.R)

        map_saf0 = (rand_map < self.p_saf0)
        map_saf1 = np.logical_and(rand_map >= self.p_saf0, rand_map < self.p_saf0 + self.p_saf1)
        map_all_faults = np.logical_or(map_saf0, map_saf1)
        
        self.map_saf0_list, self.map_saf1_list, self.map_all_faults_list = map_saf0, map_saf1, map_all_faults
        self.idx = 0
        self.map_saf0, self.map_saf1, self.map_all_faults = self.map_saf0_list[self.idx], self.map_saf1_list[self.idx], self.map_all_faults_list[self.idx]
        
        return map_saf0, map_saf1, map_all_faults
    
    def apply_fault(self, rc_code):
        # if rc_code.shape[1:] != self.map_all_faults.shape[1:]:
        #     raise ValueError(f"RC code shape ({rc_code.shape}) does not match fault map shape ({self.map_all_faults.shape})")
        # my_map_all_faults = self.map_all_faults[self.idx].reshape((1, ))
        # self.map_saf0, self.map_saf1, self.map_all_faults = self.gen_fault_map()
        faulty_code = rc_code * np.logical_not(self.map_all_faults)
        faulty_code = faulty_code + self.map_saf0*(self.q_lvl-1)
        return faulty_code
