# %%
from rc_grouping.rccodes import *
from rc_grouping.faultmaps import *
from rc_grouping.decomp_numba import *

import numpy as np
import multiprocessing as mp
from functools import partial

from typing import List
from numba import njit


def cvm_solve_multiple_parallel(
        map_all_faults_list, 
        map_saf0_list,        
        q_code_list, 
        rc_code_list, 
        calc_q_code_single_func,
        L, num_pools=20):
    
    num_batches = num_pools * 1
    num_examples = len(q_code_list)
    num_examples_per_pool = num_examples // num_batches

    start_idx = [i * num_examples_per_pool for i in range(num_batches)]
    end_idx = [(i + 1) * num_examples_per_pool for i in range(num_batches)]
    end_idx[-1] = num_examples

    args_list = [
        (
            map_all_faults_list[start_idx[i]:end_idx[i]],
            map_saf0_list[start_idx[i]:end_idx[i]],
            q_code_list[start_idx[i]:end_idx[i]]
        )
        for i in range(num_batches)
    ]

    cvm_partial = partial(
        cvm_for_list_of_q_numba,
        rc_code_list=rc_code_list,
        calc_q_code_single_func=calc_q_code_single_func,
        L = L
    )

    with mp.get_context("spawn").Pool(num_pools) as pool:
        results = pool.starmap(cvm_partial, args_list)

    cvm_rc = np.concatenate(results, axis=0)
    
    return cvm_rc


@njit(cache=True)
def cvm_for_list_of_q_numba(map_all_faults_list, map_saf0_list, q_code_list, 
                            rc_code_list, calc_q_code_single_func, L):
    num_q_codes = len(q_code_list)
    min_error_list = np.empty(num_q_codes, dtype=np.float32)
    rc_code_pair_list = np.stack((rc_code_list, rc_code_list), axis=1)
    rc_code_pair_list = rc_code_pair_list.astype(np.float32)
    final_rc_pairs_list = np.empty((num_q_codes, *rc_code_pair_list.shape[1:]), dtype=np.float32)
    
    for i in range(num_q_codes):
        my_q_code = q_code_list[i]
        map_saf0 = map_saf0_list[i]
        map_all_faults = map_all_faults_list[i]

        final_rc_pairs, min_error = cvm_for_qcode_numba(my_q_code,
                                                        rc_code_pair_list, 
                                                        map_all_faults, 
                                                        map_saf0, L,
                                                        calc_q_code_single_func)
        final_rc_pairs_list[i] = final_rc_pairs
        min_error_list[i] = min_error

    return final_rc_pairs_list


@njit(cache=True)
def cvm_for_qcode_numba(my_q_code, rc_code_pair_list, 
                        map_all_faults, map_saf0, L,
                        calc_q_code_single_func):
    # generate all possible RC codes
    faulty_code = numba_apply_fault(rc_code_pair_list, map_all_faults, map_saf0, L)

    # compute decomposition matrix between RC pairs
    num_codes = faulty_code.shape[0]
    max_q_code = calc_q_code_single_func(rc_code_pair_list[-1,0])
    # residual_matrix = np.empty((num_codes, num_codes), dtype=np.float64)
    # sparsity_matrix = np.empty((num_codes, num_codes), dtype=np.int32)
    min_residual = 2
    min_sparsity = 2*rc_code_pair_list.shape[-1]*rc_code_pair_list.shape[-2]

    for i in range(num_codes):
        for j in range(num_codes):
            pos_rc = faulty_code[i,0]
            neg_rc = faulty_code[j,1]
            pos_q = calc_q_code_single_func(pos_rc)
            neg_q = calc_q_code_single_func(neg_rc)
            decomp_q_code = pos_q - neg_q
            # decomp_rc_code = rc_code_with_faults[i] - rc_code_with_faults[j]
            # decomp_q_code = codebook._calc_q_code(decomp_rc_code)

            # compute the error with the original q_code
            q_code_residual = np.abs(decomp_q_code - my_q_code)
            q_code_residual_norm = np.sum(q_code_residual / max_q_code, dtype=np.float64)
            # residual_matrix[i,j] = q_code_residual_norm            

            if q_code_residual_norm > min_residual:
                continue

            if q_code_residual_norm < min_residual:
                min_residual = q_code_residual_norm
                min_rc_pair = np.stack((pos_rc, neg_rc))
                continue

            rc_code_sparsity = np.count_nonzero(pos_rc) + np.count_nonzero(neg_rc)
            if rc_code_sparsity < min_sparsity:
                min_sparsity = rc_code_sparsity
                min_rc_pair = np.stack((pos_rc, neg_rc))
                continue
                
    # if min_residual > 1/16:
    #     print("============================")
    #     print(f"min_residual: {min_residual}")
    #     print(f"my_q_code: {my_q_code}")
    #     pos_rc = min_rc_pair[0]
    #     neg_rc = min_rc_pair[1]
    #     pos_q = calc_q_code_single_func(pos_rc)
    #     neg_q = calc_q_code_single_func(neg_rc)
    #     decomp_q_code = pos_q - neg_q
    #     print(f"decomp_q_code: {decomp_q_code}")
    #     q_code_residual = np.abs(decomp_q_code - my_q_code)
    #     q_code_residual_norm = np.sum(q_code_residual / max_q_code, dtype=np.float64)
    #     print(f"q_code_residual: {q_code_residual}")
    #     print(f"q_code_residual_norm: {q_code_residual_norm}")

    return min_rc_pair, min_residual


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

    for my_q_code in q_code_list:
        faultmap.gen_fault_map()
        final_rc_pairs, matched = cvm_for_qcode(my_q_code, faultmap, codebook)
        matched_list.append(matched)
        if unit_test:
            residuals, min_error = unit_test_fawd(my_q_code, final_rc_pairs, faultmap, codebook)
            min_error_list.append(min_error)
            if matched and min_error != 0:
                break
        else:
            min_error_list.append(0)

        num_fault = np.sum(faultmap.map_all_faults.astype(int))
        single_fault_list.append(num_fault == 1)
        multi_fault_list.append(num_fault > 1)


    return min_error_list, matched_list, single_fault_list, multi_fault_list

def cvm_for_list_of_q_pooled(args: List):
    """Proxy function for cvm_for_list_of_q to be used in multiprocessing pool

    Args:
        args (List): input arguments in the form of List for cvm_for_list_of_q()

    Returns:
        _type_: output of cvm_for_list_of_q()
    """
    q_code_list, codebook, faultmap, decomposer, unit_test = args
    return cvm_for_list_of_q(q_code_list, codebook, faultmap, decomposer, unit_test)
