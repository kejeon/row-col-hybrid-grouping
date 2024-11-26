import faultmaps
import faultmaps_numba
import rccodes
import numpy as np

import time
from functools import wraps

def timeit(func):
    """Decorator function to measure the time taken by a function. 
    The measured time is logged using the logger of the decorated function.

    Parameters
    ----------
    func : function
        Function to be decorated.

    Returns
    -------
    wrapper : function
        Decorated function.

    Examples
    --------
    >>> @timeit
    ... def my_function():
    ...     time.sleep(1)
    ...     return
    >>> my_function()
    1994-06-01 91:06:21 - my_function - INFO - Execution time: 1.0000 seconds
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time-start_time:.4f} seconds")
        return result
    return wrapper

def test_get_fault_map():
    faultmap = faultmaps.FaultMaps(R=2, C=4, q_lvl=2, p_saf0=0.2175, p_saf1=0.3904, pos_neg_sep=True)

    start_time = time.time()
    map_saf0, map_saf1, map_all_faults = faultmap.gen_fault_map(n=10000)
    end_time = time.time()
    print(map_saf0.shape, map_saf1.shape, map_all_faults.shape)
    print(f"Execution time: {end_time-start_time:.4f} seconds")
    return

def test_get_fault_map_numba():
    faultmap = faultmaps.FaultMaps(R=2, C=4, q_lvl=2, p_saf0=0.2175, p_saf1=0.3904, pos_neg_sep=True)

    start_time = time.time()
    map_saf0, map_saf1, map_all_faults = faultmap.gen_fault_map_numba_wrapper(n=100000)
    end_time = time.time()
    # print shape of map_saf0, map_saf1, map_all_faults
    print(map_saf0.shape, map_saf1.shape, map_all_faults.shape)
    print(f"Execution time: {end_time-start_time:.4f} seconds")
    return

def test_apply_fault_numba():
    N = 100000
    codebook = rccodes.RcCodes(R=2, C=4, q_lvl=2, R_start=0, shift_base=2)
    rc_codes = codebook.rc_code
    my_rc_code_pos = rc_codes[np.random.randint(0, rc_codes.shape[0],size=N)]
    my_rc_code_neg = rc_codes[np.random.randint(0, rc_codes.shape[0],size=N)]
    my_rc_code = np.stack([my_rc_code_pos, my_rc_code_neg], axis=0)
    my_rc_code = np.swapaxes(my_rc_code, 0, 1)

    start_time = time.time()
    faultmap = faultmaps.FaultMaps(R=2, C=4, q_lvl=2, p_saf0=0.2175, p_saf1=0.3904, pos_neg_sep=True)
    map_saf0, map_saf1, map_all_faults = faultmap.gen_fault_map(n=N)
    
    faulty_code = faultmaps_numba.apply_fault_on_numpy_list_numba(my_rc_code, map_saf1, map_all_faults, q_lvl=2)
    end_time = time.time()
    print(np.mean(my_rc_code - faulty_code))
    print(f"Execution time: {end_time-start_time:.4f} seconds")
    return

def test_apply_fault():
    N = 100000
    codebook = rccodes.RcCodes(R=2, C=4, q_lvl=2, R_start=0, shift_base=2)
    rc_codes = codebook.rc_code
    my_rc_code_pos = rc_codes[np.random.randint(0, rc_codes.shape[0],size=N)]
    my_rc_code_neg = rc_codes[np.random.randint(0, rc_codes.shape[0],size=N)]
    my_rc_code = np.stack([my_rc_code_pos, my_rc_code_neg], axis=0)
    my_rc_code = np.swapaxes(my_rc_code, 0, 1)

    start_time = time.time()
    faultmap = faultmaps.FaultMaps(R=2, C=4, q_lvl=2, p_saf0=0.2175, p_saf1=0.3904, pos_neg_sep=True)

    for i in range(N):
        faultmap.gen_fault_map()
        faulty_code = faultmap.apply_fault(my_rc_code[i].reshape((1,*my_rc_code[i].shape)))
    end_time = time.time()
    # print(faulty_code)
    print(f"Execution time: {end_time-start_time:.4f} seconds")
    return

if __name__ == '__main__':
    # test_get_fault_map()
    # test_get_fault_map_numba()
    # test_get_fault_map_numba()
    test_apply_fault_numba()
    test_apply_fault_numba()
    test_apply_fault()
