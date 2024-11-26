import numpy as np
import time

# from .fawd.fawd_numba import fawd_for_list_of_q, fawd_for_list_of_q_numba
from .fawd.fawd_numba import fawd_for_list_of_q_numba_refactored as fawd_for_list_of_q_numba
from .cvm.cvm_numba import cvm_solve_multiple_parallel
from .cvx.gurobi_free import GurobiOptimizer, gurobi_solve_multiple_parallel, gurobi_solve_multiple
from .cvx.gurobi_sparsify import gurobi_solve_multiple_parallel as gurobi_sparsify_solve_multiple_parallel
from .cvx.gurobi_sparsify import gurobi_solve_multiple as gurobi_sparsify_solve_multiple

from rc_grouping.rccodes import RcCodes, gen_conversion_vector

from .cvx.gurobi_refactored import gurobi_solve_multiple as gurobi_solve_multiple_refactored
from .cvx.gurobi_refactored import gurobi_solve_multiple_parallel as gurobi_solve_multiple_parallel_refactored
from .cvx.gurobi_refactored import gurobi_fawd, gurobi_cvm_relaxed, gurobi_cvm, gurobi_cvm_int


def c_fault_free_solve_refactored(q_codes, codebook, faultmaps, decomposer_numba, num_pools):
    tic = time.time()
    if decomposer_numba.decomp_dict_rc is None:
        decomposer_numba.decomp_dict_rc = decomposer_numba._gen_decomp_dict_rc()

    C = codebook.C
    R = codebook.R
    R_start = codebook.R_start
    mem_q_lvl = codebook.L
    shift_base = codebook.b_shift

    tic_fawd = time.time()
    fawd_rc, _, fawd_matched, _, _ = fawd_for_list_of_q(q_codes, codebook, faultmaps, decomposer_numba)
    toc_fawd = time.time()
    fawd_time = toc_fawd - tic_fawd

    unmatched = np.logical_not(fawd_matched)
    unmatched_all_faults = faultmaps.map_all_faults_list[unmatched]
    unmatched_saf1 = faultmaps.map_saf1_list[unmatched]
    unmatched_saf0 = faultmaps.map_saf0_list[unmatched]
    unmatched_q_codes = q_codes[unmatched]

    rank_vec = gen_conversion_vector(C, shift_base)
    max_q_code = codebook.q_code[-1]

    tic_cvx = time.time()
    # if unmatched is not empty, solve with cvx
    if np.sum(unmatched) > 0:
        if num_pools == 1:
            cvx_rc = gurobi_solve_multiple_refactored(
                unmatched_saf1, 
                unmatched_saf0, 
                unmatched_q_codes, 
                C, R, mem_q_lvl, R_start, rank_vec, max_q_code,
                gurobi_cvm_relaxed)
        elif num_pools > 1:
            cvx_rc = gurobi_solve_multiple_parallel_refactored(
                unmatched_saf1, 
                unmatched_saf0, 
                unmatched_q_codes, 
                gurobi_cvm_relaxed,
                C, R, mem_q_lvl, R_start, rank_vec, max_q_code, num_pools=num_pools)
        else:
            raise ValueError('num_pools must be greater than 0')
        fawd_rc[unmatched] = cvx_rc
        
    toc_cvx = time.time()
    cvx_time = toc_cvx - tic_cvx
    total_time = toc_cvx - tic

    stats = {
        'num_q_codes': len(q_codes),
        'num_matched': np.sum(fawd_matched),
        'num_unmatched': len(q_codes) - np.sum(fawd_matched),
        'perc_matched': float(np.sum(fawd_matched) / len(q_codes))*100,
        'fawd_time': fawd_time,
        'cvx_time': cvx_time,
        'total_time': total_time
    }

    return fawd_rc, fawd_matched, stats

def c_fault_free_solve(q_codes, codebook, faultmaps, decomposer_numba, num_pools):
    tic = time.time()
    if decomposer_numba.decomp_dict_rc is None:
        decomposer_numba.decomp_dict_rc = decomposer_numba._gen_decomp_dict_rc()

    C = codebook.C
    R = codebook.R
    R_start = codebook.R_start
    mem_q_lvl = codebook.L
    shift_base = codebook.b_shift

    tic_fawd = time.time()
    fawd_rc, _, fawd_matched, _, _ = fawd_for_list_of_q(q_codes, codebook, faultmaps, decomposer_numba)
    toc_fawd = time.time()
    fawd_time = toc_fawd - tic_fawd

    unmatched = np.logical_not(fawd_matched)
    unmatched_all_faults = faultmaps.map_all_faults_list[unmatched]
    unmatched_saf0 = faultmaps.map_saf0_list[unmatched]
    unmatched_saf1 = faultmaps.map_saf1_list[unmatched]
    unmatched_q_codes = q_codes[unmatched]

    rank_vec = gen_conversion_vector(C, shift_base)
    max_q_code = codebook.q_code[-1]

    tic_cvx = time.time()
    # if unmatched is not empty, solve with cvx
    if np.sum(unmatched) > 0:
        if num_pools == 1:
            cvx_rc = gurobi_solve_multiple(
                unmatched_all_faults, 
                unmatched_saf0, 
                unmatched_q_codes, 
                GurobiOptimizer,
                C, R, mem_q_lvl, R_start, rank_vec, max_q_code)
        elif num_pools > 1:
            cvx_rc = gurobi_solve_multiple_parallel(
                unmatched_all_faults, 
                unmatched_saf0, 
                unmatched_q_codes, 
                GurobiOptimizer,
                C, R, mem_q_lvl, R_start, rank_vec, max_q_code, num_pools=num_pools)
        else:
            raise ValueError('num_pools must be greater than 0')
        fawd_rc[unmatched] = cvx_rc
        
    toc_cvx = time.time()
    cvx_time = toc_cvx - tic_cvx
    total_time = toc_cvx - tic

    stats = {
        'num_q_codes': len(q_codes),
        'num_matched': np.sum(fawd_matched),
        'num_unmatched': len(q_codes) - np.sum(fawd_matched),
        'perc_matched': float(np.sum(fawd_matched) / len(q_codes))*100,
        'fawd_time': fawd_time,
        'cvx_time': cvx_time,
        'total_time': total_time
    }

    return fawd_rc, fawd_matched, stats

def c_fault_free_adv_solve(q_codes, codebook, faultmaps, decomposer_numba, num_pools, bypass_fawd=False):
    C = codebook.C
    R = codebook.R
    R_start = codebook.R_start
    mem_q_lvl = codebook.L
    shift_base = codebook.b_shift
    sig_vec = codebook.sig_vec
    max_q_code = codebook.q_code[-1]
    saf0_list = faultmaps.map_saf0_list
    saf1_list = faultmaps.map_saf1_list
    safa_list = faultmaps.map_all_faults_list
    final_rc = np.zeros_like(saf0_list, dtype=np.int32)

    tic = time.time()
    
    faulty_offset = np.einsum('k,ijkl->ji', sig_vec, saf0_list*(mem_q_lvl-1))
    faulty_offset = faulty_offset[0] - faulty_offset[1]
    faulty_range = np.einsum('k,ijkl->ji', sig_vec, np.logical_not(safa_list)*(mem_q_lvl-1))
    faulty_max = faulty_offset + faulty_range[0]
    faulty_min = faulty_offset - faulty_range[1]

    # check if q_codes are within the faulty range
    over = np.greater(q_codes[:,0], faulty_max)
    under = np.less(q_codes[:,0], faulty_min)
    out_of_range = np.logical_or(over, under)
    in_range = np.logical_not(out_of_range)

    # if q_codes are out of range, set them to the nearest boundary
    final_rc[over,0] = np.logical_not(safa_list[over,0])*(mem_q_lvl-1)
    final_rc[under,1] = np.logical_not(safa_list[under,1])*(mem_q_lvl-1)

    toc_range = time.time()

    # check if q_codes are continuously representable
    safa_not_list = np.logical_not(safa_list)
    colwise_no_faults = np.einsum('ijkl->ijk', safa_not_list)
    colwise_all_faults = np.logical_not(colwise_no_faults)
    col_all_faults = np.all(colwise_all_faults, axis=1)
    col_all_faults = np.any(col_all_faults, axis=1)
    col_no_faults = np.logical_not(col_all_faults)

    toc_cont = time.time()

    # if q_codes are within range and continuously representable, solve with fawd
    cont_representable = np.logical_and(in_range, col_no_faults)
    if not bypass_fawd:
        if decomposer_numba is not None:
            if decomposer_numba.decomp_dict_rc is None:
                decomposer_numba.decomp_dict_rc = decomposer_numba._gen_decomp_dict_rc()
            # fawd_rc, _, _, _, _ = fawd_for_list_of_q(
            #     q_codes, codebook, faultmaps, decomposer_numba)
            map_saf0_list = faultmaps.map_saf0_list[cont_representable]
            map_saf1_list = faultmaps.map_saf1_list[cont_representable]
            L = codebook.L
            decomp_dict = decomposer_numba.decomp_dict_rc
            q_code_list = q_codes[cont_representable]
            fawd_rc, _, _, _, _ = fawd_for_list_of_q_numba(
                q_code_list, 
                map_saf0_list, 
                map_saf1_list, 
                L, 
                decomp_dict)
        else:
            if num_pools == 1:
                fawd_rc = gurobi_solve_multiple_refactored(
                            saf1_list[cont_representable], 
                            saf0_list[cont_representable], 
                            q_codes[cont_representable], 
                            C, R, mem_q_lvl, R_start, sig_vec, max_q_code,
                            gurobi_fawd)
            elif num_pools > 1:
                fawd_rc = gurobi_solve_multiple_parallel_refactored(
                            saf1_list[cont_representable], 
                            saf0_list[cont_representable], 
                            q_codes[cont_representable], 
                            C, R, mem_q_lvl, R_start, sig_vec, max_q_code,
                            gurobi_fawd, num_pools=num_pools)
        final_rc[cont_representable] = fawd_rc
    
    toc_sparsity = time.time()

    # apply cvm on remaining q_codes
    remaining = np.logical_and(in_range, col_all_faults)
    if np.sum(remaining) != 0:
        if num_pools == 1:
            cvm_rc = gurobi_solve_multiple_refactored(
                    saf1_list[remaining], 
                    saf0_list[remaining], 
                    q_codes[remaining],
                    C, R, mem_q_lvl, R_start, sig_vec, max_q_code, 
                    gurobi_cvm_relaxed)
        elif num_pools > 1:
            cvm_rc = gurobi_solve_multiple_parallel_refactored(
                    saf1_list[remaining], 
                    saf0_list[remaining], 
                    q_codes[remaining],
                    C, R, mem_q_lvl, R_start, sig_vec, max_q_code,
                    gurobi_cvm_relaxed, num_pools=num_pools)
        final_rc[remaining] = cvm_rc
    toc_cvm = time.time()
    # faultmaps.map_all_faults_list = faultmaps.map_all_faults_list[in_range]
    # faultmaps.map_saf0_list = faultmaps.map_saf0_list[in_range]
    # faultmaps.map_saf1_list = faultmaps.map_saf1_list[in_range]
    # cff_rc, fawd_matched, stats = c_fault_free_solve(
    #     q_codes[in_range], codebook, faultmaps, decomposer_numba, num_pools)
    # final_rc[in_range] = cff_rc

    num_in_range = np.sum(in_range)
    num_out_of_range = np.sum(out_of_range)
    num_cont_representable = np.sum(cont_representable)
    num_remaining = np.sum(remaining)

    stats = {
        'num_q_codes': len(q_codes),
        'vec_in_range': in_range,
        'vec_out_of_range': out_of_range,
        'vec_cont_representable': cont_representable,
        'vec_remaining': remaining,
        'num_in_range': num_in_range,
        'num_out_of_range': num_out_of_range,
        'num_cont_representable': num_cont_representable,
        'num_remaining': num_remaining,
        'range_time': toc_range - tic,
        'cont_time': toc_cont - toc_range,
        'sparsity_time': toc_sparsity - toc_cont,
        'cvm_time': toc_cvm - toc_sparsity,
        'total_time': toc_cvm - tic
    }


    # debug_vec = {
    #     'out_of_range': out_of_range,
    #     'faulty_offset': faulty_offset,
    #     'faulty_max': faulty_max,
    #     'faulty_min': faulty_min,
    # }

    return final_rc, None, stats

def fault_free_solve(q_codes, codebook, faultmaps, decomposer_numba, num_pools):
    tic = time.time()

    C = codebook.C
    R = codebook.R
    R_start = codebook.R_start
    mem_q_lvl = codebook.L
    shift_base = codebook.b_shift

    tic_fawd = time.time()
    fawd_rc, _, fawd_matched, _, _ = fawd_for_list_of_q(q_codes, codebook, faultmaps, decomposer_numba)
    toc_fawd = time.time()
    fawd_time = toc_fawd - tic_fawd

    unmatched = np.logical_not(fawd_matched)
    unmatched_all_faults = faultmaps.map_all_faults_list[unmatched]
    unmatched_saf0 = faultmaps.map_saf0_list[unmatched]
    unmatched_q_codes = q_codes[unmatched]

    rank_vec = gen_conversion_vector(C, shift_base)
    max_q_code = codebook.q_code[-1]

    tic_cvm = time.time()
    cvm_rc = cvm_solve_multiple_parallel(
        unmatched_all_faults,
        unmatched_saf0,
        unmatched_q_codes,
        codebook.rc_code,
        codebook.calc_q_code_single,
        mem_q_lvl,
        num_pools=num_pools
    )
    toc_cvm = time.time()
    cvm_time = toc_cvm - tic_cvm

    fawd_rc[unmatched] = cvm_rc

    toc = time.time()
    total_time = toc - tic

    stats = {
        'perc_matched': float(np.sum(fawd_matched) / len(q_codes))*100,
        'fawd_time': fawd_time,
        'cvm_time': cvm_time,
        'total_time': total_time
    }

    return fawd_rc, fawd_matched, stats