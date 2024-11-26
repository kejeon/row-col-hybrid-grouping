import numpy as np
from numba import njit

def gen_conversion_vector(C, shift_base):
    vec = shift_base**np.arange(C)
    # Must be in float32 or float64; numba doesnt support int matmul
    vec = vec[::-1].astype(np.float32) 
    return vec

@njit(cache=True)
def calc_q_code_single_inside(rc_code, vec, R_start):
    q_code =  vec @ rc_code
    q_code = np.cumsum(q_code)
    q_code = q_code[R_start:].astype(np.int32)
    return q_code

@njit(cache=True)
def calc_q_code_list_inside(rc_code_array, vec, R_start):
    rc_code_shape = rc_code_array.shape    
    len_rc_code = rc_code_shape[0]

    rc_code = rc_code_array[0]
    q_code = calc_q_code_single_inside(rc_code, vec, R_start)
    out = np.empty((len_rc_code, *q_code.shape), dtype=np.int32)
    out[0] = q_code

    for i in range(1,len_rc_code):
        rc_code = rc_code_array[i]
        # print(rc_code)
        q_code = calc_q_code_single_inside(rc_code, vec, R_start)
        out[i] = q_code
    return out

@njit()
def calc_q_code_array_inside(rc_code_array, vec, R_start):
    rc_code_shape = rc_code_array.shape    
    len_rc_code_1 = rc_code_shape[0]
    len_rc_code_2 = rc_code_shape[1]
    # print(rc_code_array.shape)
    # print(rc_code_array.dtype)

    rc_code = rc_code_array[0,0]
    q_code = calc_q_code_single_inside(rc_code, vec, R_start)
    out = np.empty((len_rc_code_1, len_rc_code_2, *q_code.shape), dtype=np.int32)

    for i in range(0,len_rc_code_1):
        for j in range(0,len_rc_code_2):
            rc_code = rc_code_array[i,j]
            # print(rc_code)  
            q_code = calc_q_code_single_inside(rc_code, vec, R_start)
            out[i,j] = q_code

    return out

def calc_q_code_gen_func(inner_func, C, shift_base, R_start):
    vec = gen_conversion_vector(C, shift_base)
    
    @njit(cache=True)
    def calc_q_code(rc_code):
        return inner_func(rc_code, vec, R_start)

    return calc_q_code


def gen_cartesian_prod(A, B):
  cartesian = []

  for i in range(len(A)):
    for j in range(len(B)):
      temp = np.vstack((A[i], B[j]))
      cartesian.append(temp)

  cartesian = np.array(cartesian)

  return cartesian

def calc_code_vals(code, C, shift_base):
  code_shape = code.shape
  code_vals = np.sum(code, axis=len(code_shape)-1)
  if C==1:
    return code_vals

  exponent = np.flip(np.array(range(C)))
  exp_val = shift_base**exponent
  code_vals = code_vals@exp_val

  return code_vals

def calc_q_code(code, R_start, shift_base):
    R = code.shape[-1]
    C = code.shape[-2]

    out = None

    for r in range(R_start,R):
        temp = code[:, 0:C, 0:r+1]
        vals = calc_code_vals(temp, C=C, shift_base=shift_base).flatten()
        if out is None:
            out = vals
        else:
            out = np.vstack((out,vals))

    out = out.T
    
    if len(out.shape) == 1:
        out = out.reshape(-1,1)

    return out

def gen_codes_R(R, q_lvl):
    alphabets = list(range(q_lvl))
    code_matrix = None

    for i in range(0, R):
        my_row = np.array([])
        for j in alphabets:
            temp_row = np.array([j]*(q_lvl**i))
            my_row = np.concatenate((my_row, temp_row))
        my_row = np.tile(my_row, (1, q_lvl**(R - i - 1)))
        if code_matrix is None:
            code_matrix = my_row
        else:
            code_matrix = np.vstack((my_row, code_matrix))

    code_matrix = code_matrix.T

    # calculate the values of each code. Shift base doesnt matter here
    code_vals = calc_code_vals(code=code_matrix, C=1, shift_base=None)

    return code_matrix, code_vals

def gen_codes_RC(R, C, q_lvl, shift_base):
    C1_code_matrix, code_vals = gen_codes_R(R=R, q_lvl=q_lvl)
    if C==1:
        return C1_code_matrix.reshape(C1_code_matrix.shape[0],1,-1), code_vals

    # generate an array containing all combinations of vectors from C1_code_matrix
    code_tensor = C1_code_matrix
    for i in range(C-1):
        code_tensor = gen_cartesian_prod(C1_code_matrix, code_tensor)

    code_vals = calc_code_vals(code=code_tensor, C=C, shift_base=shift_base)

    return code_tensor, code_vals

def gen_q_code_emp(code, q_code, R_start, R, C, q_lvl, 
                     shift_base, my_weights=100000,
                     q_mode='trunc'):
    if type(my_weights) is int:
        my_weights = np.random.rand(my_weights)
    elif type(my_weights) is np.ndarray:
        my_weights = my_weights
    else:
       raise ValueError('Unsupported input for my_weights')
    
    final_codes = []

    for i in range(R_start, R):
        num_q_lvl = calc_q_lvl_RC(R=i+1, C=C, 
                                  mem_q_lvl=q_lvl, 
                                  shift_base=shift_base)
        if q_mode == 'trunc':
            temp = np.trunc(my_weights * num_q_lvl)
        elif q_mode == 'round':
            temp = np.round(my_weights * (num_q_lvl-1))
        else:
           raise ValueError("q_mode not supported")
        
        final_codes.append(temp)
    
    emp_q_codes = np.array(final_codes).T
    uniq_codes = np.unique(emp_q_codes, axis=0)

    return uniq_codes, emp_q_codes

def calc_q_lvl_RC(R, C, mem_q_lvl, shift_base):
    q_row = 1 + R*(mem_q_lvl - 1)
    if C==1:
        return q_row

    exponent = np.array(range(C))
    exp_val = shift_base**exponent
    num_q_lvl = 1 + (q_row - 1)*np.sum(exp_val)

    return num_q_lvl

def get_rc_code_emp(rc_code_all, q_code_all, q_code_emp):
    rc_code_emp = np.empty([1, rc_code_all.shape[1], 
                            rc_code_all.shape[2]])

    for x in q_code_emp:
        # check if x is in q_code_all
        my_idx = np.all(q_code_all == x, axis=1)
        my_rc_code = rc_code_all[my_idx]

        if my_idx.sum() == 0:
            print('Error: %d not found' %x)
            break 
        rc_code_emp = np.concatenate((rc_code_emp, my_rc_code))

    rc_code_emp = rc_code_emp[1:,:,:]

    return rc_code_emp

def gen_rc_code_dict(q_code_emp, rc_code_emp):
    my_dict = {}
    for i in range(q_code_emp.shape[0]):
        my_key = str(q_code_emp[i])
        if my_key in my_dict:
            my_dict[my_key].append(rc_code_emp[i])
        else:
            my_dict[my_key] = [rc_code_emp[i]]
            
    
    return my_dict

def get_q_range_val(q_val, num_q_level):
    # Note that this function is for trunc quantization not round quantization
    step = 1/(num_q_level)
    lb = q_val*step
    ub = q_val*step + step
    return [lb, ub]

def get_q_range(code, base):
  my_range = []

  for i in range(len(code)):
    temp = get_q_range_val(q_val=code[i],
                           num_q_level=base[i])
    my_range.append(temp)

  lb = 0
  ub = 1

  for i in my_range:
    if i[0] > lb:
      lb = i[0]
    if i[1] < ub:
      ub = i[1]

  return np.array([lb, ub])

def get_all_q_range(q_code_emp):
    all_q_range = np.zeros((1, 2))
    for i in q_code_emp:
        temp = get_q_range(code=i, base=q_code_emp[-1]+1)
        all_q_range = np.vstack((all_q_range, temp))
    return all_q_range[1:]

class RcCodes():
    def __init__(self, R_start, R, C, q_lvl, shift_base):
        super().__init__()
        self.R_start = R_start  # R_start is the starting value of row truncation.
                                # If R_start = 0, that means the only first row is not truncatable
                                # If R_start = 1, that means all rows starting from the second row are truncatable
        self.R = R
        self.C = C
        self.L = q_lvl
        self.b_shift = shift_base
        self.sig_vec = gen_conversion_vector(
            C=self.C, shift_base=self.b_shift)
        self._check_input()
        
        # generate codebook
        self.rc_code, self.q_code = self._generate_codes()
        self.calc_q_code_single = calc_q_code_gen_func(inner_func= calc_q_code_single_inside,
                                                        C=self.C, 
                                                        shift_base=self.b_shift, 
                                                        R_start=self.R_start)
        self.calc_q_code_list = calc_q_code_gen_func(inner_func= calc_q_code_list_inside,
                                                    C=self.C, 
                                                    shift_base=self.b_shift, 
                                                    R_start=self.R_start)
        self.calc_q_code_array = calc_q_code_gen_func(inner_func= calc_q_code_array_inside,
                                                    C=self.C, 
                                                    shift_base=self.b_shift, 
                                                    R_start=self.R_start)
        

    def _check_input(self):
        if self.R_start >= self.R:
            raise ValueError('R_start must be less than R')
        self._check_shift_base()
        return

    def _check_shift_base(self):
        if self.b_shift < 2:
            raise ValueError('Shift base must be greater than 2')
        elif self.b_shift > (self.L-1)*(self.R_start+1) + 1:
            raise ValueError('Shift base must be less than %d' %((self.L-1)*(self.R_start+1) + 1))
        return

    def _generate_codes(self):
        # generate all possible RC codeㄴ
        rc_code_all, _ = gen_codes_RC(R=self.R, C=self.C, q_lvl=self.L, 
                                      shift_base=self.b_shift)
        # generate corresponding Q-codes (values for different degrees of trucncation)
        q_code_all = calc_q_code(code=rc_code_all, 
                                 R_start=self.R_start, 
                                 shift_base=self.b_shift)
        # some of the codes generated above may be illegal; cleand the codes by using random weights.
        rand_weight = np.random.rand(1000000)
        q_code_emp, _ = gen_q_code_emp(code=rc_code_all, q_code=q_code_all, 
                                        R_start=self.R_start, R=self.R, 
                                        C=self.C, q_lvl=self.L, 
                                        shift_base=self.b_shift, 
                                        my_weights=rand_weight, 
                                        q_mode='trunc')
        
        # get the corresponding RC codes for the cleaned Q-codes
        rc_code_emp = get_rc_code_emp(rc_code_all, q_code_all, q_code_emp).astype(int)
        q_code_emp = calc_q_code(code=rc_code_emp,
                                R_start=self.R_start,
                                shift_base=self.b_shift).astype(int)

        rc_code = rc_code_emp
        q_code = q_code_emp

        return rc_code, q_code
    
    def _calc_q_code(self, my_rc_code):
       return calc_q_code(code=my_rc_code, 
                          R_start=self.R_start, 
                          shift_base=self.b_shift)

    def print_trunc_comb(self):
        for i in range(1,self.R + 1):
            for j in range(1,self.C + 1):
                num_q = self._calc_q_lvl_RC(R_t=i, C_t=j)
                print("%d,%d: %d" %(i,j,num_q))


