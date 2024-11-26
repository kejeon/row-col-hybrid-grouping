import numpy as np
from numba import njit

@njit
def np_any_axis123(x):
    out = np.zeros(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[0]):
        if np.sum(x[i], dtype=np.int8) > 0:
            out[i] = True
        else:
            out[i] = False
    return out

@njit
def np_count_nonzero_axis123(x):
    out = np.zeros(x.shape[0], dtype=np.int8)
    for i in range(x.shape[0]):
        out[i] = np.count_nonzero(x[i])
    return out

@njit
def getitems_with_bool_vectors(my_array, bool_vector):
    # check the number of Trues in a bool vector
    num_trues = np.count_nonzero(bool_vector)
    if num_trues == 0:
        return None
    # create a new array with the same shape as the bool vector
    my_array_shape = my_array.shape
    out_array_shape = (num_trues,) + my_array_shape[1:]
    out_array = np.zeros(out_array_shape, dtype=my_array.dtype)

    count = 0
    for i in range(my_array_shape[0]):
        if bool_vector[i]:
            out_array[count] = my_array[i]
            count += 1
            if count == num_trues:
                return out_array