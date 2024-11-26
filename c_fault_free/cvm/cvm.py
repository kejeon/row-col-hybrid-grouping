# %%
from rc_grouping.rccodes import *
from rc_grouping.faultmaps import *
from rc_grouping.decomp import *

import numpy as np
from multiprocessing import Pool

from typing import Type, Tuple, List
import time
import os
from utils.utils import set_timezone, setup_log_dir, create_logger

set_timezone('Asia/Seoul')
log_dir = setup_log_dir('logs')
logger = create_logger('fawd', log_dir)


def fawd_for_list_of_q(q_code_list: List[np.ndarray],
                        codebook: RcCodes,
                        faultmap: FaultMaps,
                        decomposer: Decomp,
                        unit_test: bool
                        ) -> Tuple[List[bool], List[bool], List[int], List[int]]:

    min_error_list = []
    matched_list = []
    single_fault_list = []
    multi_fault_list = []

    for my_q_code in q_code_list:
        faultmap.gen_fault_map()
        final_rc_pairs, matched = fawd_for_qcode(my_q_code, faultmap, codebook, decomposer)
        matched_list.append(matched)
        if not matched:
            # get the index of my_q_code from codebook.q_code
            rc_pairs = decomposer.get_decomp_pairs_rc(my_q_code)
            final_rc_pairs = rc_pairs[np.random.randint(len(rc_pairs))]
            final_rc_pairs = np.array([final_rc_pairs])
            
        if unit_test:
            residuals, min_error = unit_test_fawd(my_q_code, final_rc_pairs, faultmap, codebook)
            min_error_list.append(min_error)
            if matched and min_error != 0:
                logger.error("Fault-free unit test failed: FAWD match found but error is non-zero")
                break
        else:
            min_error_list.append(0)

        num_fault = np.sum(faultmap.map_all_faults.astype(int))
        single_fault_list.append(num_fault == 1)
        multi_fault_list.append(num_fault > 1)

    return min_error_list, matched_list, single_fault_list, multi_fault_list

def fawd_for_list_of_q_pooled(args: List):
    """Proxy function for fawd_for_list_of_q to be used in multiprocessing pool

    Args:
        args (List): input arguments in the form of List for fawd_for_list_of_q()

    Returns:
        _type_: output of fawd_for_list_of_q()
    """
    q_code_list, codebook, faultmap, decomposer, unit_test = args
    return fawd_for_list_of_q(q_code_list, codebook, faultmap, decomposer, unit_test)

def cvm_for_list_of_q(q_code_list: List[np.ndarray],
                        codebook: RcCodes,
                        faultmap: FaultMaps,
                        decomposer: Decomp,
                        unit_test: bool
                        ) -> Tuple[List[bool], List[bool], List[int], List[int]]:

    min_error_list = []
    matched_list = []
    single_fault_list = []
    multi_fault_list = []
    final_rc_pairs_list = []

    for my_q_code in q_code_list:
        faultmap.next()
        final_rc_pairs, matched = cvm_for_qcode(my_q_code, faultmap, codebook)
        final_rc_pairs_list.append(final_rc_pairs)
        matched_list.append(matched)
        if unit_test:
            residuals, min_error = unit_test_fawd(my_q_code, final_rc_pairs, faultmap, codebook)
            min_error_list.append(min_error)
            if matched and min_error != 0:
                logger.error("Fault-free unit test failed: FAWD match found but error is non-zero")
                break
        else:
            min_error_list.append(0)

        num_fault = np.sum(faultmap.map_all_faults.astype(int))
        single_fault_list.append(num_fault == 1)
        multi_fault_list.append(num_fault > 1)


    return final_rc_pairs_list, min_error_list, matched_list, single_fault_list, multi_fault_list

def cvm_for_list_of_q_pooled(args: List):
    """Proxy function for cvm_for_list_of_q to be used in multiprocessing pool

    Args:
        args (List): input arguments in the form of List for cvm_for_list_of_q()

    Returns:
        _type_: output of cvm_for_list_of_q()
    """
    q_code_list, codebook, faultmap, decomposer, unit_test = args
    return cvm_for_list_of_q(q_code_list, codebook, faultmap, decomposer, unit_test)

def fault_free_for_list_of_q(q_code_list: List[np.ndarray],
                             codebook: RcCodes,
                             faultmap: FaultMaps,
                             decomposer: Decomp,
                             unit_test: bool
                             ) -> Tuple[List[bool], List[bool], List[int], List[int]]:
    min_error_list = []
    matched_list = []
    single_fault_list = []
    multi_fault_list = []

    for my_q_code in q_code_list:
        faultmap.gen_fault_map()
        final_rc_pairs, matched = fawd_for_qcode(my_q_code, faultmap, codebook, decomposer)
        matched_list.append(matched)

        # if not matched, run cvm
        if not matched:
            final_rc_pairs, min_val = cvm_for_qcode(my_q_code, faultmap, codebook)

        if unit_test:
            residuals, min_error = unit_test_fawd(my_q_code, final_rc_pairs, faultmap, codebook)
            min_error_list.append(min_error)
            if matched and min_error != 0:
                logger.error("Fault-free unit test failed: FAWD match found but error is non-zero")
                break
        else:
            min_error_list.append(0)

        # collect stats on faults
        num_fault = np.sum(faultmap.map_all_faults.astype(int))
        single_fault_list.append(num_fault == 1)
        multi_fault_list.append(num_fault > 1)

    return min_error_list, matched_list, single_fault_list, multi_fault_list

def fault_free_for_list_of_q_pooled(args: List):
    """Proxy function for fault_free_for_list_of_q to be used in multiprocessing pool

    Args:
        args (List): input arguments in the form of List for fault_free_for_list_of_q()

    Returns:
        _type_: output of fault_free_for_list_of_q()
    """
    q_code_list, codebook, faultmap, decomposer, unit_test = args
    return fault_free_for_list_of_q(q_code_list, codebook, faultmap, decomposer, unit_test)

def unit_test_fawd(my_q_code: np.ndarray, 
                   rc_pairs: np.ndarray, 
                   faultmap: FaultMaps, 
                   codebook: RcCodes
                   ) -> Tuple[np.ndarray, np.float64]:
    """Compute the error between a given Q-code and a given RC-code pairs. 
    To achieve this, we first apply the faults from the faultmap to the RC pairs
    to generate the corrupted RC-pairs. Then, we compute the Q-code of the 
    corrupted RC-pairs and compute the error with the given Q-code. 
    
    If FAWD pair exists, the error should be 0. 

    Parameters
    ----------
    my_q_code : Numpy array of int
        Q-code to be compared with.
    rc_pairs : Numpy array of int
        RC-pairs to be tested with.
    faultmap : _type_
        Fault map that will be imposed on the given RC-pairs.
    codebook : _type_
        Codebook class that encloses all possible Q-codes and RC-codes for a given configuration.

    Returns
    -------
    residuals: Numpy array of fp64
        Error between the given Q-code and the Q-code of the corrupted RC-pairs.
    min_error: Numpy fp64
        Minimum error between the given Q-code and the Q-code of the corrupted RC-pairs.
    """
    # generate the rc_pairs corrupted by the faultmap
    rc_pairs_corrupted = faultmap.apply_fault(rc_pairs)
    pos_q = codebook._calc_q_code(rc_pairs_corrupted[:,0])
    neg_q = codebook._calc_q_code(rc_pairs_corrupted[:,1])
    max_q_code = codebook._calc_q_code(codebook.rc_code[np.newaxis, -1])

    final_q = pos_q - neg_q

    residuals = np.abs(final_q - my_q_code) / max_q_code
    residuals = np.sum(residuals, axis=-1)

    min_error = np.amin(residuals)

    return residuals, min_error

def fawd_for_qcode(my_q_code, faultmap, codebook, decomposer):
    # STEP 1: Get all decomposition combinations
    # Generate all decompositions
    rc_pairs = decomposer.get_decomp_pairs_rc(my_q_code)
    rc_pairs = np.array(rc_pairs)

    # STEP 2: Filter the combinations based on the faultmap
    # Generate boolean vectors for the locations of logical zero and one
    not_zero = rc_pairs != 0 # False for logical 0 True for the rest
    not_one = rc_pairs != (codebook.L - 1) # False for logical 1 True for the rest
    # Note that logical_zero_locs and logical_one_locs are not necessarily the inverse of each other 
    # (for values between 0 and 1, in the case of mulit-level cells, they are always true)

    # Generate boolean vectors that will be false if the saf is not masked, and true if saf is masked
    saf0_masked = np.logical_and(not_zero, np.array(faultmap.map_saf0))
    saf1_masked = np.logical_and(not_one, np.array(faultmap.map_saf1))
    saf0_masked_idx = np.any(saf0_masked, axis=(1,2,3))
    saf1_masked_idx = np.any(saf1_masked, axis=(1,2,3))
    # The saf0 and saf1 should be masked
    fault_masked_idx = np.logical_and(np.logical_not(saf0_masked_idx), 
                                      np.logical_not(saf1_masked_idx))
    
    # Filter the rc_pairs based on the fault map
    final_rc_pairs = rc_pairs[fault_masked_idx]

    # STEP 3: Return the RC pairs and the matched boolean
    # If no decomposition exists, return empty array and matched = False
    if final_rc_pairs.shape[0] == 0:
        return final_rc_pairs, False

    # If decomposition exists, return the sparsest RC pair
    one_count = np.sum(final_rc_pairs, axis=tuple(range(1, final_rc_pairs.ndim)))
    idx = one_count == np.amin(one_count)
    final_rc_pairs = final_rc_pairs[idx]
    return final_rc_pairs, True

def cvm_for_qcode(my_q_code, faultmap, codebook):
    # generate all possible RC codes
    rc_codes = codebook.rc_code
    rc_pairs = np.stack((codebook.rc_code, codebook.rc_code), axis=1)
    rc_code_with_faults = faultmap.apply_fault(rc_pairs)

    # compute decomposition matrix between RC pairs
    num_codes = rc_codes.shape[0]
    decomp_matrix = np.zeros((num_codes, num_codes))
    max_q_code = codebook._calc_q_code(rc_codes[np.newaxis, -1])


    for i in range(num_codes):
        for j in range(num_codes):
            pos_q = codebook._calc_q_code(rc_code_with_faults[np.newaxis,i,0])
            neg_q = codebook._calc_q_code(rc_code_with_faults[np.newaxis,j,1])
            decomp_q_code = pos_q - neg_q
            # decomp_rc_code = rc_code_with_faults[i] - rc_code_with_faults[j]
            # decomp_q_code = codebook._calc_q_code(decomp_rc_code)

            # compute the error with the original q_code
            q_code_residual = np.abs(decomp_q_code - my_q_code)
            decomp_matrix[i,j] = np.sum(q_code_residual / max_q_code)

    # find the closest values
    # return all smallest values from the decomp_matrix
    min_val = np.amin(decomp_matrix)
    min_idx = np.where(decomp_matrix == min_val)
    min_rc_codes = []
    min_sparsity = 2*codebook.R*codebook.C

    (x, y) = min_idx
    for i, j in zip(x,y):
        my_rc_pair = np.stack((rc_codes[i], rc_codes[j]))
        # check sparsity
        nonzero_count = np.count_nonzero(my_rc_pair)
        if nonzero_count < min_sparsity:
            min_sparsity = nonzero_count
            min_rc_codes.append(my_rc_pair)

    return np.array(min_rc_codes[0]), min_val


