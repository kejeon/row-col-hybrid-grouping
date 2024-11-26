import numpy as np
from numba import njit
from numba.typed import Dict
from numba.core import types

@njit(nogil=True, cache=True)
def _qcode2idx(q_code):
    qcode_len = len(q_code)
    multiplier = 10**5 ## TODO: should be specific to RC code dim
    idx = 0
    for i in range(qcode_len):
        new_idx = 500 + q_code[i]
        idx += new_idx
        idx *= multiplier
    
    return int(idx)


@njit(nogil=True, cache=True)
def _get_decomp_pairs_rc(decomp_dict_rc, q_code):
    q_code_str = _qcode2idx(q_code)
    
    # catch key errors
    if q_code_str in decomp_dict_rc:
        decomp_pairs_rc = decomp_dict_rc[q_code_str]
    else:
        raise ValueError('No decomposition pairs found for the given q_code') 
    
    return decomp_pairs_rc


class Decomp():
    def __init__(self, rc_codebook):
        super().__init__()
        self.rc_codebook = rc_codebook
        self._q_codes = rc_codebook.q_code.astype(np.int16)
        self._rc_codes = rc_codebook.rc_code.astype(np.int16)

        # generate hashmap for decomposition
        self.decomp_dict = self._gen_decomp_dict()
        self.decomp_dict_rc = self._gen_decomp_dict_rc()

    def _gen_decomp_dict(self):
        q_codes = self._q_codes
        decomp_dict = {}
        for i in range(len(q_codes)):
            for j in range(len(q_codes)):
                decomp_val = q_codes[i] - q_codes[j]
                decomp_key = _qcode2idx(decomp_val)
                if decomp_key in decomp_dict:
                    decomp_dict[decomp_key].append((i,j))
                else:
                    decomp_dict[decomp_key] = [(i,j)]
        return decomp_dict
    
    def get_decomp_pairs_idx(self, q_code):
        q_code_str = _qcode2idx(q_code)
        decomp_dict = self.decomp_dict

        # catch key errors
        if q_code_str in decomp_dict:
            decomp_pairs_idx = decomp_dict[q_code_str]
        else:
            raise ValueError('No decomposition pairs found for the given q_code') 
        
        return decomp_pairs_idx
    
    def _gen_decomp_dict_rc(self):
        q_codes = self._q_codes
        q_codes = np.concatenate((q_codes, -q_codes[1:]))

        rc_codes = self._rc_codes
        # decomp_dict = {}
        decomp_dict = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.int16, 4, 'C')
        )
        for i in range(len(q_codes)):
            q_code = q_codes[i]
            q_code_str = _qcode2idx(q_code)
            decomp_pairs_idx = self.get_decomp_pairs_idx(q_code)
            num_pairs = len(decomp_pairs_idx)
            temp_array = np.zeros((num_pairs, 2, *self._rc_codes.shape[1:]), dtype=np.int16)
            # print(temp_array.shape)
            for j in range(num_pairs):
                temp_array[j][0] = rc_codes[decomp_pairs_idx[j][0]]
                temp_array[j][1] = rc_codes[decomp_pairs_idx[j][1]]

            sparsity = np.count_nonzero(temp_array, axis=(1,2,3))
            sort_idx = np.argsort(sparsity)
            sorted_temp_array = temp_array[sort_idx]
            decomp_dict[q_code_str] = sorted_temp_array

        return decomp_dict

    
    def get_decomp_pairs_rc(self, q_code):
        return _get_decomp_pairs_rc(self.decomp_dict_rc, q_code)
    