import numpy as np
from numba import jit, njit, vectorize

def gen_fault_map_numba(C, R, p_saf0, p_saf1, pos_neg_sep, n=1):
    rand_map = np.random.rand(n, C, R)
    if pos_neg_sep:
        rand_map = np.random.rand(n, 2, C, R)
    else:
        rand_map = np.random.rand(n, 1, C, R)

    map_saf0 = (rand_map < p_saf0).astype(np.int8)
    map_saf1 = np.logical_and(rand_map >= p_saf0, rand_map < p_saf0 + p_saf1, dtype=np.int8)
    map_all_faults = np.logical_or(map_saf0, map_saf1, dtype=np.int8)
        
    return map_saf0, map_saf1, map_all_faults

@njit
def apply_fault_numba(rc_code, map_saf1, map_all_faults, q_lvl):
    faulty_code = rc_code * np.logical_not(map_all_faults)
    faulty_code = faulty_code + map_saf1*(q_lvl-1)
    return faulty_code

def apply_fault_on_numpy_list_numba(rc_codes, map_saf1, map_all_faults, q_lvl):
    N = map_saf1.shape[0]
    out = np.zeros(rc_codes.shape)
    for i in range(N):
        out[i] = apply_fault_numba(rc_codes[i], map_saf1[i], map_all_faults[i], q_lvl)
    return out

class FaultMaps():
    def __init__(self, 
                 R: int, 
                 C: int, 
                 q_lvl: int, 
                 n:int=1, 
                 p_saf0:float=0.0175, 
                 p_saf1:float=0.0904, 
                 pos_neg_sep:bool=True):
        super().__init__()
        self.R = R
        self.C = C
        self.q_lvl = q_lvl
        self.p_saf0 = p_saf0
        self.p_saf1 = p_saf1
        self.pos_neg_sep = pos_neg_sep
        self.p_no_saf = 1 - p_saf0 - p_saf1

        self.map_saf0, self.map_saf1, self.map_all_faults = self.gen_fault_map(n)
        self.idx = 0

    def gen_fault_map(self, n=1, use_numba=False):
        if use_numba:
            numba_func = njit()(gen_fault_map_numba)
            return numba_func(self.C, self.R, self.p_saf0, self.p_saf1, self.pos_neg_sep, n)
        
        return gen_fault_map_numba(self.C, self.R, self.p_saf0, self.p_saf1, self.pos_neg_sep, n)

    def apply_fault(self, rc_codes, use_numba=True):
        if use_numba:
            numba_func = njit()(apply_fault_on_numpy_list_numba)
            return numba_func(rc_codes, self.map_saf1, self.map_all_faults, self.q_lvl)

        return apply_fault_numba(rc_codes, self.map_saf1, self.map_all_faults, self.q_lvl)