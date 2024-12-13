from rc_grouping.rccodes import *
from rc_grouping.faultmaps import *
from rc_grouping.decomp_numba import *
from rc_grouping.decomp_numba import _get_decomp_pairs_rc

import numpy as np
from numba import njit

@njit(cache=True)
def numba_cond_check_at_idx(idx, array, val):
    a = idx[0]
    b = idx[1]
    c = idx[2]
    num_idx = a.shape[0]

    for i in range(num_idx):
        if array[a[i], b[i], c[i]] != val:
            return False
    
    return True


@njit(cache=True)
def fawd_for_qcode_numba(my_q_code, map_saf0, map_saf1, L, rc_pairs):
    num_pairs = rc_pairs.shape[0]
    map_saf0_idx = np.where(map_saf0==1)
    map_saf1_idx = np.where(map_saf1==1)   

    for i in range(0, num_pairs):
        rc_code = rc_pairs[i]
        # Check if the saf1 is masked
        saf1_masked = numba_cond_check_at_idx(map_saf1_idx, rc_code, 0)
        if not saf1_masked:
            continue
        # Check if the saf0 is masked
        saf0_masked = numba_cond_check_at_idx(map_saf0_idx, rc_code, L-1)
        if not saf0_masked:
            continue
        return np.expand_dims(rc_code, axis=0), True

    # if count == 0:
    return np.expand_dims(rc_pairs[0], axis=0), False


@njit(cache=True)
def fawd_for_list_of_q_numba(q_code_list, map_saf0_list, map_saf1_list, L,
                              decomp_dict):
    num_q_codes = len(q_code_list)
    matched_list = np.empty(num_q_codes, dtype=np.bool8)
    min_error_list = None
    single_fault_list = None
    multi_fault_list = None
    final_rc_pairs_list = np.empty_like(map_saf0_list, dtype=np.int32)

    for i in range(num_q_codes):
        my_q_code = q_code_list[i]
        map_saf0 = map_saf0_list[i]
        map_saf1 = map_saf1_list[i]
        # check if my_q_code is negative
        rc_pairs = _get_decomp_pairs_rc(decomp_dict, my_q_code)

        final_rc_pairs, matched = fawd_for_qcode_numba(my_q_code, map_saf0, map_saf1, L, rc_pairs)
        matched_list[i] = matched
        final_rc_pairs_list[i] = final_rc_pairs
        
    return final_rc_pairs_list, min_error_list, matched_list, single_fault_list, multi_fault_list


def fawd_for_list_of_q(q_code_list, codebook, faultmap, decomposer):
    map_saf0_list = faultmap.map_saf0_list
    map_saf1_list = faultmap.map_saf1_list
    L = codebook.L
    decomp_dict = decomposer.decomp_dict_rc
    return fawd_for_list_of_q_numba(q_code_list, map_saf0_list, map_saf1_list, L, decomp_dict)

