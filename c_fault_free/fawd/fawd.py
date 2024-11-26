# %%
from rc_grouping.rccodes import *
from rc_grouping.faultmaps import *
from rc_grouping.decomp_numba import *

from utils.utils import set_timezone, setup_log_dir, create_logger

import numpy as np
from multiprocessing import Pool
import time
import os
from typing import Type, Tuple, List



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
        final_rc_pairs, matched = fawd_for_qcode(my_q_code, faultmap, codebook, decomposer)
        matched_list.append(matched)
        if not matched:
            # get the index of my_q_code from codebook.q_code
            rc_pairs = decomposer.get_decomp_pairs_rc(my_q_code)
            final_rc_pairs = rc_pairs[np.random.randint(len(rc_pairs))]
            final_rc_pairs = np.array([final_rc_pairs])
            
        # if unit_test:
        #     residuals, min_error = unit_test_fawd(my_q_code, final_rc_pairs, faultmap, codebook)
        #     min_error_list.append(min_error)
        #     if matched and min_error != 0:
        #         logger.error("Fault-free unit test failed: FAWD match found but error is non-zero")
        #         break
        # else:
        #     min_error_list.append(0)

        num_fault = np.sum(faultmap.map_all_faults.astype(int))
        single_fault_list.append(num_fault == 1)
        multi_fault_list.append(num_fault > 1)

        faultmap.next()


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
        return np.expand_dims(rc_pairs[0], axis=0), False

    # If decomposition exists, return the sparsest RC pair
    one_count = np.sum(final_rc_pairs, axis=tuple(range(1, final_rc_pairs.ndim)))
    idx = one_count == np.amin(one_count)
    final_rc_pairs = final_rc_pairs[idx]
    final_rc_pairs = final_rc_pairs[0]
    final_rc_pairs = np.expand_dims(final_rc_pairs, axis=0)
    
    return final_rc_pairs, True


