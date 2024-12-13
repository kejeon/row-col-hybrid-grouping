import numpy as np
from rc_grouping.faultmaps import FaultMaps, numba_apply_fault

def compute_faulty_q_code(rc_code, all_faults, saf0, codebook):
    faulty_code = numba_apply_fault(
        rc_code, 
        all_faults, 
        saf0, 
        codebook.L)
    q_code_fawd = codebook.calc_q_code_list(faulty_code.astype(np.float32))
    q_code_fawd = q_code_fawd[0] - q_code_fawd[1]
    return q_code_fawd

def compute_residual(q_code, q_code_faulty, max_q_code):
    residual = np.sum(np.abs(q_code - q_code_faulty) / max_q_code)
    return residual

def compute_residual_multiple(q_codes, rc_codes, faultmaps, codebook):
    num_code = len(q_codes)
    residuals = np.zeros(num_code)
    residual_total = 0
    max_q_code = codebook.q_code[-1]

    for i in range(num_code):
        q_code = q_codes[i]
        rc_code = rc_codes[i]
        all_faults = faultmaps.map_all_faults_list[i]
        saf0 = faultmaps.map_saf0_list[i]

        q_code_faulty = compute_faulty_q_code(rc_code, all_faults, saf0, codebook)
        residual = compute_residual(q_code, q_code_faulty, max_q_code)
        residuals[i] = residual
        residual_total += residual

    residual_avg = residual_total / (num_code * len(max_q_code))

    return residuals, residual_avg
