import torch
import numpy as np
import pandas as pd
from dnn_model.models.quan_ops import Conv2d_Q

def ks_smd_compute_performance_by_module(
        module, 
        imc_array_dim, 
        num_row_grouping, num_col_grouping, 
        act_count=None, **kwargs):
    if not isinstance(module, Conv2d_Q):
        raise ValueError('Module must be nn.Conv2d')
    
    out_channel = module.out_channels
    in_channel = module.in_channels
    kernel_size = module.kernel_size[0]

    num_pe = kernel_size ** 2
    num_mapping_rows = in_channel * num_row_grouping        
    num_mapping_cols = out_channel * num_col_grouping

    smd_ar_repeat = np.floor(imc_array_dim / num_mapping_rows)
    smd_ac_repeat = np.floor(imc_array_dim / num_mapping_cols)
    smd_multiplier = np.minimum(smd_ar_repeat, smd_ac_repeat)
    if smd_multiplier == 0:
        # print("Submatrix duplication is not feasible")
        smd_multiplier = 1
        ar_cycle = np.ceil(num_mapping_rows / imc_array_dim)
        ac_cycle = np.ceil(num_mapping_cols / imc_array_dim)
    else:
        # smd is only feasible when the number of mapping rows and cols are less than imc_array_dim
        ar_cycle = 1
        ac_cycle = 1
    
    total_cycle = ar_cycle * ac_cycle * num_pe

    throughput = smd_multiplier / total_cycle
    # cycle_per_op = 1 / throughput_per_cycle
    total_cycle_act = None
    if act_count is not None:
        total_cycle_act = act_count / throughput

    imc_array_utilization = (num_mapping_rows * num_mapping_cols)*smd_multiplier / (imc_array_dim ** 2) / (ar_cycle * ac_cycle)

    return smd_multiplier, total_cycle, total_cycle_act, throughput, imc_array_utilization, num_mapping_rows, num_mapping_cols

def ks_smd_compute_performance_by_model(
        model, 
        imc_array_dim, 
        RC_config,
        prec_config,
        act_count_keys,
        act_count_vals):
    df = pd.DataFrame(columns=['layer_name', 'smd_multiplier', 'total_cycle', 'throughput', 'imc_array_utilization', 'num_mapping_rows', 'num_mapping_cols', 'num_row_grouping', 'num_col_grouping'])
    counter_rc_config = 0
    counter_prec_config = 0
    total_cycle = 0
    num_param = 0
    avg_prec = 0

    for name, module in model.named_modules():
        if not isinstance(module, Conv2d_Q):
            continue
        if type(RC_config) is dict:
            my_RC_config = RC_config
        elif type(RC_config) is list:
            my_RC_config = RC_config[counter_rc_config]
            counter_rc_config += 1
        else:
            raise ValueError('RC_config must be a list or a dictionary')
        
        if type(prec_config) is int:
            my_prec_config = prec_config
        elif type(prec_config) is list:
            my_prec_config = prec_config[counter_prec_config]
            counter_prec_config += 1
        else:
            raise ValueError('prec_config must be an integer or a list')
        
        rho, cycle, throughput, util, num_rows, num_cols = ks_smd_compute_performance_by_module(module, imc_array_dim, **my_RC_config)

        act_count_idx = [k in name for k in act_count_keys]
        act_count = np.array(act_count_vals)[act_count_idx]
        current_cycle = act_count / throughput
        total_cycle += current_cycle

        current_num_param = module.weight.numel()
        num_param += current_num_param
        avg_prec += my_prec_config * current_num_param

        hw_perf = {
            'layer_name': name,
            'smd_multiplier': rho, 
            'total_cycle': cycle, 
            'throughput': throughput, 
            'imc_array_utilization': util, 
            'num_mapping_rows': num_rows, 
            'num_mapping_cols': num_cols}
        
        hw_perf.update(my_RC_config)

        df = df.append(hw_perf, 
            ignore_index=True)

    if counter_rc_config != 0 and counter_rc_config != len(RC_config):
        raise ValueError('RC_config list length does not match the number of Conv2d_Q modules in the model')

    if counter_prec_config != 0 and counter_prec_config != len(prec_config):
        raise ValueError('prec_config list length does not match the number of Conv2d_Q modules in the model')

    avg_prec /= num_param

    return df, total_cycle, avg_prec

def identify_best_RC_config(RC_config_list, 
                            config_prec_list,
                            imc_array_dim,
                            model,
                            act_dim_dict):
    # reorganize the lists
    prec_idx = np.argsort(config_prec_list)[::-1]
    config_prec_list = np.array(config_prec_list)[prec_idx]
    RC_config_list = np.array(RC_config_list)[prec_idx]

    # find the best config
    best_config = []
    best_prec = []
    layer_name_list = []
    num_cycle_list = []

    for name, module in model.named_modules():
        if isinstance(module, Conv2d_Q):
            tp_list = []
            act_dim = act_dim_dict[name]
            # act dim is a tuple. get a multiple of all of its elements
            act_dim = np.prod(act_dim)
            for config in RC_config_list:
                rho, cycle, total_cycle_act, throughput, util, num_rows, num_cols = ks_smd_compute_performance_by_module(module, imc_array_dim, **config)
                tp_list.append(throughput)
                
            best_idx = np.argmax(tp_list)
            max_tp = tp_list[best_idx]
            num_cycle = act_dim / max_tp
            num_cycle_list.append(num_cycle)
            best_config.append(RC_config_list[best_idx])
            best_prec.append(config_prec_list[best_idx])
            layer_name_list.append(name)

    return best_config, best_prec, layer_name_list, num_cycle_list
