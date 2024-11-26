# %%
from rc_grouping.rccodes import *
from rc_grouping.faultmaps import *
from rc_grouping.decomp_numba import *
from rc_grouping.decomp_numba import _qcode2idx, _get_decomp_pairs_rc

import numpy as np
import multiprocessing as mp
from functools import partial

from typing import Type, Tuple, List
from utils.utils import set_timezone, setup_log_dir, create_logger

from numba import jit, njit, prange
from ..utils.numba_utils import np_any_axis123, np_count_nonzero_axis123, getitems_with_bool_vectors


@njit(cache=True)
def fawd_for_qcode_numba(my_q_code, map_saf0, map_saf1, L, rc_pairs):
    # STEP 2: Filter the combinations based on the faultmap
    # Generate boolean vectors for the locations of logical zero and one
    not_zero = rc_pairs != 0 # False for logical 0 True for the rest
    not_one = rc_pairs != (L - 1) # False for logical 1 True for the rest
    # Note that logical_zero_locs and logical_one_locs are not necessarily the inverse of each other 
    # (for values between 0 and 1, in the case of mulit-level cells, they are always true)

    # Generate boolean vectors that will be false if the saf is not masked, and true if saf is masked
    saf0_masked = np.logical_and(not_zero, map_saf0)
    saf1_masked = np.logical_and(not_one, map_saf1)
    saf0_masked_idx = np_any_axis123(saf0_masked)
    saf1_masked_idx = np_any_axis123(saf1_masked)
    # The saf0 and saf1 should be masked
    fault_masked_idx = np.logical_and(np.logical_not(saf0_masked_idx), 
                                      np.logical_not(saf1_masked_idx))
    # STEP 3: Return the RC pairs and the matched boolean
    # Filter the rc_pairs based on the fault map
    # If no decomposition exists, return empty array and matched = False
    final_rc_pairs = getitems_with_bool_vectors(rc_pairs, fault_masked_idx)
    if final_rc_pairs is None:
        return np.expand_dims(rc_pairs[0], axis=0), False
    
    # If decomposition exists, return the sparsest RC pair
    one_count = np_count_nonzero_axis123(final_rc_pairs)
    idx = one_count == np.amin(one_count)

    final_rc_pairs = getitems_with_bool_vectors(final_rc_pairs, idx)
    final_rc_pairs = final_rc_pairs[0]
    final_rc_pairs = np.expand_dims(final_rc_pairs, axis=0)

    return final_rc_pairs, True


def fawd_for_qcode(my_q_code, faultmap, codebook, decomposer):
    map_saf0 = faultmap.map_saf0
    map_saf1 = faultmap.map_saf1
    L = codebook.L
    rc_pairs = decomposer.get_decomp_pairs_rc(my_q_code)
    return fawd_for_qcode_numba(my_q_code, map_saf0, map_saf1, L, rc_pairs)


# @njit(cache=True)
# def fawd_for_qcode_numba2(my_q_code, map_saf0, map_saf1, L, rc_pairs):
#     not_one = rc_pairs != (L - 1) # False for logical 1 True for the rest
#     saf1_masked = np.logical_and(not_one, map_saf1)
#     saf1_masked_idx = np_any_axis123(saf1_masked)
#     final_rc_pairs = getitems_with_bool_vectors(rc_pairs, np.logical_not(saf1_masked_idx))

#     if final_rc_pairs is None:
#         return np.expand_dims(rc_pairs[0], axis=0), False

#     not_zero = final_rc_pairs != 0
#     saf0_masked = np.logical_and(not_zero, map_saf0)
#     saf0_masked_idx = np_any_axis123(saf0_masked)
#     final_rc_pairs = getitems_with_bool_vectors(final_rc_pairs, np.logical_not(saf0_masked_idx))

#     if final_rc_pairs is None:
#         return np.expand_dims(rc_pairs[0], axis=0), False
    
#     # If decomposition exists, return the sparsest RC pair
#     one_count = np_count_nonzero_axis123(final_rc_pairs)
#     idx = one_count == np.amin(one_count)

#     final_rc_pairs = getitems_with_bool_vectors(final_rc_pairs, idx)
#     final_rc_pairs = final_rc_pairs[0]
#     final_rc_pairs = np.expand_dims(final_rc_pairs, axis=0)

#     return final_rc_pairs, True


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
def fawd_for_qcode_numba3(my_q_code, map_saf0, map_saf1, L, rc_pairs):
    num_pairs = rc_pairs.shape[0]
    map_saf0_idx = np.where(map_saf0==1)
    map_saf1_idx = np.where(map_saf1==1)
    out = np.empty(num_pairs, dtype=np.int32)
    count = 0

    # implementation of the faulted weight extraction method
    rc_code = rc_pairs[0]
    saf0_masked = numba_cond_check_at_idx(map_saf0_idx, rc_code, L-1)
    saf1_masked = numba_cond_check_at_idx(map_saf1_idx, rc_code, 0)
    if saf1_masked and saf0_masked:
        return np.expand_dims(rc_pairs[0], axis=0), True        

    for i in range(1, num_pairs):
        rc_code = rc_pairs[i]
        # Check if the saf1 is masked
        saf1_masked = numba_cond_check_at_idx(map_saf1_idx, rc_code, 0)
        if not saf1_masked:
            continue
        # Check if the saf0 is masked
        saf0_masked = numba_cond_check_at_idx(map_saf0_idx, rc_code, L-1)
        if not saf0_masked:
            continue
        out[count] = i
        count += 1

    if count == 0:
        return np.expand_dims(rc_pairs[0], axis=0), False
    
    final_rc_pairs = rc_pairs[out[:count]]

    # If decomposition exists, return the sparsest RC pair
    one_count = np_count_nonzero_axis123(final_rc_pairs)
    idx = one_count == np.amin(one_count)

    final_rc_pairs = getitems_with_bool_vectors(final_rc_pairs, idx)
    final_rc_pairs = final_rc_pairs[0]
    final_rc_pairs = np.expand_dims(final_rc_pairs, axis=0)

    return final_rc_pairs, True


@njit(cache=True)
def fawd_for_qcode_numba4(my_q_code, map_saf0, map_saf1, L, rc_pairs):
    num_pairs = rc_pairs.shape[0]
    map_saf0_idx = np.where(map_saf0==1)
    map_saf1_idx = np.where(map_saf1==1)
    # out = np.empty(num_pairs, dtype=np.int32)
    # count = 0

    # implementation of the faulted weight extraction method
    # rc_code = rc_pairs[0]
    # saf0_masked = numba_cond_check_at_idx(map_saf0_idx, rc_code, L-1)
    # saf1_masked = numba_cond_check_at_idx(map_saf1_idx, rc_code, 0)
    # if saf1_masked and saf0_masked:
    #     return np.expand_dims(rc_pairs[0], axis=0), True        

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
    
    # final_rc_pairs = rc_pairs[out[:count]]

    # # If decomposition exists, return the sparsest RC pair
    # one_count = np_count_nonzero_axis123(final_rc_pairs)
    # idx = one_count == np.amin(one_count)

    # final_rc_pairs = getitems_with_bool_vectors(final_rc_pairs, idx)
    # final_rc_pairs = final_rc_pairs[0]
    # final_rc_pairs = np.expand_dims(final_rc_pairs, axis=0)

    # return final_rc_pairs, True


@njit(cache=True)
def fawd_for_list_of_q_numba(q_code_list, map_saf0_list, map_saf1_list, L,
                              decomp_dict, unit_test=False):
    num_q_codes = len(q_code_list)
    matched_list = np.empty(num_q_codes, dtype=np.bool8)
    min_error_list = np.empty(num_q_codes, dtype=np.int32)
    single_fault_list = np.empty(num_q_codes, dtype=np.bool8)
    multi_fault_list = np.empty(num_q_codes, dtype=np.bool8)
    final_rc_pairs_list = np.empty_like(map_saf0_list, dtype=np.int32)

    for i in range(num_q_codes):
        my_q_code = q_code_list[i]
        map_saf0 = map_saf0_list[i]
        map_saf1 = map_saf1_list[i]
        # check if my_q_code is negative
        rc_pairs = _get_decomp_pairs_rc(decomp_dict, my_q_code)

        final_rc_pairs, matched = fawd_for_qcode_numba3(my_q_code, map_saf0, map_saf1, L, rc_pairs)
        matched_list[i] = matched
        final_rc_pairs_list[i] = final_rc_pairs
        
        if not matched:
            # get the index of my_q_code from codebook.q_code
            final_rc_pairs = rc_pairs[np.random.randint(len(rc_pairs))]
            
        # if unit_test:
        #     residuals, min_error = unit_test_fawd(my_q_code, final_rc_pairs, faultmap, codebook)
        #     min_error_list.append(min_error)
        #     if matched and min_error != 0:
        #         logger.error("Fault-free unit test failed: FAWD match found but error is non-zero")
        #         break
        # else:
        #     min_error_list.append(0)

        num_fault = np.sum(map_saf0) + np.sum(map_saf1)
        single_fault_list[i] = num_fault == 1
        multi_fault_list[i] = num_fault > 1
    return final_rc_pairs_list, min_error_list, matched_list, single_fault_list, multi_fault_list


@njit(cache=True)
def fawd_for_list_of_q_numba_refactored(q_code_list, map_saf0_list, map_saf1_list, L,
                              decomp_dict):
    num_q_codes = len(q_code_list)
    matched_list = np.empty(num_q_codes, dtype=np.bool8)
    min_error_list = None
    single_fault_list = None
    multi_fault_list = None
    # min_error_list = np.empty(num_q_codes, dtype=np.int32)
    # single_fault_list = np.empty(num_q_codes, dtype=np.bool8)
    # multi_fault_list = np.empty(num_q_codes, dtype=np.bool8)
    final_rc_pairs_list = np.empty_like(map_saf0_list, dtype=np.int32)

    for i in range(num_q_codes):
        my_q_code = q_code_list[i]
        map_saf0 = map_saf0_list[i]
        map_saf1 = map_saf1_list[i]
        # check if my_q_code is negative
        rc_pairs = _get_decomp_pairs_rc(decomp_dict, my_q_code)

        final_rc_pairs, matched = fawd_for_qcode_numba4(my_q_code, map_saf0, map_saf1, L, rc_pairs)
        matched_list[i] = matched
        final_rc_pairs_list[i] = final_rc_pairs
        
        # if not matched:
        #     # get the index of my_q_code from codebook.q_code
        #     final_rc_pairs = rc_pairs[np.random.randint(len(rc_pairs))]
            
        # if unit_test:
        #     residuals, min_error = unit_test_fawd(my_q_code, final_rc_pairs, faultmap, codebook)
        #     min_error_list.append(min_error)
        #     if matched and min_error != 0:
        #         logger.error("Fault-free unit test failed: FAWD match found but error is non-zero")
        #         break
        # else:
        #     min_error_list.append(0)

        # num_fault = np.sum(map_saf0) + np.sum(map_saf1)
        # single_fault_list[i] = num_fault == 1
        # multi_fault_list[i] = num_fault > 1
        
    return final_rc_pairs_list, min_error_list, matched_list, single_fault_list, multi_fault_list

# @njit(cache=True)
# def nothing_for_list_of_q_numba(q_code_list, map_saf0_list, map_saf1_list, L,
#                               decomp_dict, unit_test=False):
#     num_q_codes = len(q_code_list)
#     matched_list = np.empty(num_q_codes, dtype=np.bool8)
#     min_error_list = np.empty(num_q_codes, dtype=np.int32)
#     single_fault_list = np.empty(num_q_codes, dtype=np.bool8)
#     multi_fault_list = np.empty(num_q_codes, dtype=np.bool8)
#     final_rc_pairs_list = np.empty_like(map_saf0_list, dtype=np.int32)

#     for i in range(num_q_codes):
#         my_q_code = q_code_list[i]
#         map_saf0 = map_saf0_list[i]
#         map_saf1 = map_saf1_list[i]
#         # check if my_q_code is negative
#         rc_pairs = _get_decomp_pairs_rc(decomp_dict, my_q_code)

#         final_rc_pairs, matched = rc_pairs[0], True
#         matched_list[i] = matched
#         final_rc_pairs_list[i] = final_rc_pairs
            
#         num_fault = np.sum(map_saf0) + np.sum(map_saf1)
#         single_fault_list[i] = num_fault == 1
#         multi_fault_list[i] = num_fault > 1
#     return final_rc_pairs_list, min_error_list, matched_list, single_fault_list, multi_fault_list



def fawd_for_list_of_q(q_code_list, codebook, faultmap, decomposer, unit_test=False):
    num_qcode = len(q_code_list)
    # faultmap.gen_fault_map(num_qcode)
    map_saf0_list = faultmap.map_saf0_list
    map_saf1_list = faultmap.map_saf1_list
    L = codebook.L
    decomp_dict = decomposer.decomp_dict_rc
    return fawd_for_list_of_q_numba(q_code_list, map_saf0_list, map_saf1_list, L, decomp_dict, unit_test)


def fawd_for_list_of_q_refactored(q_code_list, codebook, faultmap, decomposer, unit_test=False):
    num_qcode = len(q_code_list)
    # faultmap.gen_fault_map(num_qcode)
    map_saf0_list = faultmap.map_saf0_list
    map_saf1_list = faultmap.map_saf1_list
    L = codebook.L
    decomp_dict = decomposer.decomp_dict_rc
    return fawd_for_list_of_q_numba_refactored(q_code_list, map_saf0_list, map_saf1_list, L, decomp_dict)


# def nothing_for_list_of_q(q_code_list, codebook, faultmap, decomposer, unit_test=False):
#     num_qcode = len(q_code_list)
#     # faultmap.gen_fault_map(num_qcode)
#     map_saf0_list = faultmap.map_saf0_list
#     map_saf1_list = faultmap.map_saf1_list
#     L = codebook.L
#     decomp_dict = decomposer.decomp_dict_rc
#     return nothing_for_list_of_q_numba(q_code_list, map_saf0_list, map_saf1_list, L, decomp_dict, unit_test)



def fawd_for_list_of_q_pooled(args: List):
    """Proxy function for fawd_for_list_of_q to be used in multiprocessing pool

    Args:
        args (List): input arguments in the form of List for fawd_for_list_of_q()

    Returns:
        _type_: output of fawd_for_list_of_q()
    """
    q_code_list, codebook, faultmap, decomposer, unit_test = args
    return fawd_for_list_of_q(q_code_list, codebook, faultmap, decomposer, unit_test)


# %%
