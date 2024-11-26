# %%
import numpy as np

class Decomp():
    def __init__(self, rc_codebook):
        super().__init__()
        self.rc_codebook = rc_codebook
        self._q_codes = rc_codebook.q_code
        self._rc_codes = rc_codebook.rc_code

        # generate hashmap for decomposition
        self.decomp_dict = self._gen_decomp_dict()
        self.decomp_dict_rc = self._gen_decomp_dict_rc()
    
    
    def _gen_decomp_dict(self):
        q_codes = self._q_codes
        decomp_dict = {}
        for i in range(len(q_codes)):
            for j in range(len(q_codes)):
                decomp_val = q_codes[i] - q_codes[j]
                decomp_key = str(decomp_val)
                if decomp_key in decomp_dict:
                    decomp_dict[decomp_key].append((i,j))
                else:
                    decomp_dict[decomp_key] = [(i,j)]
        return decomp_dict
    

    def get_decomp_pairs_idx(self, q_code):
        q_code_str = str(q_code)
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
        decomp_dict = {}
        for i in range(len(q_codes)):
            q_code = q_codes[i]
            q_code_str = str(q_code)
            decomp_pairs_idx = self.get_decomp_pairs_idx(q_code)
            num_pairs = len(decomp_pairs_idx)
            temp_array = np.zeros((num_pairs, 2, *self._rc_codes.shape[1:]))
            for j in range(num_pairs):
                temp_array[j][0] = rc_codes[decomp_pairs_idx[j][0]]
                temp_array[j][1] = rc_codes[decomp_pairs_idx[j][1]]
            decomp_dict[q_code_str] = temp_array
        return decomp_dict


    def get_decomp_pairs_rc(self, q_code):
        q_code_str = str(q_code)
        decomp_dict_rc = self.decomp_dict_rc

        # catch key errors
        if q_code_str in decomp_dict_rc:
            decomp_pairs_rc = decomp_dict_rc[q_code_str]
        else:
            raise ValueError('No decomposition pairs found for the given q_code') 
        
        return decomp_pairs_rc
    
# %%
