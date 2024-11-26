import torch
import torch.nn as nn
import torch.optim

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_start_method

from cycle_calc import identify_best_RC_config

from rc_grouping.rccodes import RcCodes
from rc_grouping.faultmaps import FaultMaps
from rc_grouping.decomp_numba import Decomp
from c_fault_free.c_fault_free import c_fault_free_solve, fault_free_solve
from c_fault_free.c_fault_free import c_fault_free_adv_solve as c_fault_free_solve
# from c_fault_free.utils.test_utils import compute_faulty_q_code, compute_residual_multiple

from utils.nn_utils import get_layer_by_name, set_precision
from utils.nn_utils import quantize_conv2d, q_freeze_conv2d, q_unfreeze_conv2d
from utils.exp_utils import print_config, setup_exp, iterate_all_configs, elaborate_configs
from utils.exp_utils import *
from utils.nn_data import act_dim_dict, load_act_dim_data
import utils.exp_utils as exp_utils

from utils.utils import set_timezone, setup_log_dir, create_logger, save_dict, load_dict
from utils.utils import setup_dir, logger_info, logger_newline, logger_bold, logger_lvl1, logger_lvl2, green

# To prevent training from crashing due to truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

forward_eval = None
forward_adap = None
exp_result_path = None

def main():
    global forward_eval, forward_adap, exp_result_path
    # set exp configurations
    nn_config = exp_utils.base_nn_config.copy()
    nn_config['model'] = 'resnet18q'
    nn_config['num_workers'] = 16
    nn_config['lead_gpu_idx'] = 0
    nn_config['all_gpu_idx'] = [0]
    # nn_config = load_dict('./exp_results/resnet20qe1__2024-09-11_17-48-10/nn_config.yaml')
    act_dim_dict = load_act_dim_data()
    act_dim_dict = act_dim_dict[nn_config['model']]

    imc_config = exp_utils.base_imc_config.copy()
    imc_config['num_parallel_processes'] = 1
    imc_config['cell_precision'] = 4
    imc_config['imc_array_dim'] = 64
    iter_configs = {
        'rc_config': [
            [R1C4],
            # [R2C2],
            # [R2C4],
        ]
    }

    imc_config_list = iterate_all_configs(imc_config, iter_configs)
    imc_config_list = elaborate_configs(imc_config_list)

    exp_result_path = f"./exp_results_compile_time/{nn_config['model']}_"
    exp_result_path = setup_dir(exp_result_path)
    
    set_timezone()
    logger = create_logger('exp_compile_time', exp_result_path)
    logger_bold(f"Setting up experiment result directory...")    
    logger_lvl1(f"- Experiment result directory: {green(exp_result_path)}")

    logger_newline()
    logger_bold(f"Using nn_config")
    print_config(nn_config)

    logger_newline()
    logger_bold(f"Using iter_config")
    print_config(iter_configs)

    # save nn_config as yaml file
    save_dict(nn_config, f"{exp_result_path}/nn_config.yaml")
    save_dict(iter_configs, f"{exp_result_path}/iter_configs.yaml")

    logger_newline()
    logger_bold(f"Received {green(len(imc_config_list))} imc_config")
    logger_newline()

    model, load_train, load_val, f_eval, f_adap = setup_exp(nn_config)
    forward_eval = f_eval
    forward_adap = f_adap

    logger_newline()
    logger_bold(f"=====================================")
    logger_bold(f"======= COMPILATION TIME EXP ========")
    logger_bold(f"=====================================")
    for idx, imc_config in enumerate(imc_config_list):
        logger_bold(f"Using imc_config {idx+1}/{len(imc_config_list)}")
        run_path = f"{exp_result_path}/run_{idx}"
        run_path = setup_dir(run_path)
        stats = run_exp(
            imc_config, model, act_dim_dict=act_dim_dict, exp_idx=idx, num_trial=1, ada_bn_repeat=1,
            val_loader=load_val, train_loader=load_train, run_path=run_path)
        # logger_lvl1(f"Top1 mean: {top1_mean:.2f}%, Top1 std: {top1_std:.2f}%")
        save_dict(stats, f"{run_path}/final_result.yaml")

    return

def run_exp(imc_config, model, act_dim_dict, num_trial, val_loader, train_loader=None, ada_bn_repeat=1, exp_idx=0, run_path=None):
    if run_path is None:
        run_path = setup_dir(f"{exp_result_path}/run_{exp_idx}")

    # save imc_config as yaml file
    save_dict(imc_config, f"{run_path}/imc_config.yaml")

    logger_newline()
    print_config(imc_config)
    best_config, best_prec, layer_name_list, num_cycle_list = identify_best_RC_config(
        imc_config['rc_config'], 
        imc_config['rc_config_prec'], 
        imc_config['imc_array_dim'], 
        model,
        act_dim_dict)
    layer_prec = [(layer_name_list[i], best_prec[i]) for i in range(len(layer_name_list))]
    layer_config = [(layer_name_list[i], best_config[i]) for i in range(len(layer_name_list))]
    total_cycle = np.sum(num_cycle_list)

    # save best config as yaml file
    layer_config_dict = {}
    for idx, layer_name in enumerate(layer_name_list):
        temp_dict = {}
        temp_dict['rc_config'] = best_config[idx]
        temp_dict['precision'] = int(best_prec[idx])
        temp_dict['num_cycle'] = float(num_cycle_list[idx])
        layer_config_dict[layer_name] = temp_dict
    save_dict(layer_config_dict, f"{run_path}/layer_config.yaml")

    # update the precision of the DNN model with the identified precision configuration
    set_precision(model, layer_prec)
    quantize_conv2d(model)
    q_freeze_conv2d(model)

    codebook_dict = {}
    decomposer_dict = {}
    faultmaps_dict = {}

    for rc_config in imc_config['rc_config']:
        name = rc_config['name']
        num_row_grouping = rc_config['num_row_grouping']
        num_col_grouping = rc_config['num_col_grouping']
        r_start = rc_config['r_start']
        codebook = RcCodes(
            R_start=r_start,
            R=num_row_grouping, 
            C=num_col_grouping,
            q_lvl=imc_config['cell_precision'],
            shift_base=imc_config['shift_base']
        )
        decomposer = Decomp(rc_codebook=codebook)
        faultmaps = FaultMaps(
            R=num_row_grouping,
            C=num_col_grouping,
            q_lvl=imc_config['cell_precision'],
            p_saf0=imc_config['p_saf0'],
            p_saf1=imc_config['p_saf1'],
            pos_neg_sep=True
        )
        codebook_dict[name] = codebook
        decomposer_dict[name] = decomposer
        faultmaps_dict[name] = faultmaps

    total_hist = None

    for (layer_name, best_prec) in layer_prec:
        layer = get_layer_by_name(model, layer_name)
        w_q_int = layer.w_q_int
        # get histogram of w_q_int
        hist, bin_edges = np.histogram(w_q_int.detach().cpu().numpy(), bins=np.array(range(-best_prec, best_prec)) + 0.5)
        if total_hist is None:
            total_hist = hist
        else:
            total_hist += hist

    # generate random np.array based of total_hist
    total_hist = total_hist / np.sum(total_hist)
    total_hist = np.cumsum(total_hist)
    total_hist = np.insert(total_hist, 0, 0)
    total_hist[-1] = 1
    np.random.seed(42)

    num_params = 11_173_962
    random_array = np.random.rand(num_params)
    # random_array = np.random.rand(1_000_000)
    random_array = np.digitize(random_array, total_hist)
    random_array = random_array - best_prec
    random_vector = random_array.reshape(-1,1).astype(np.int64)

    config_name = rc_config['name']
    my_faultmaps = faultmaps_dict[config_name]
    my_faultmaps.gen_fault_map(random_vector.shape[0])
    # with ProcessPoolExecutor(max_workers=imc_config['num_parallel_processes']) as executor:
    #     result = executor.submit(
    #             inject_noise_single, 
    #             layer_name, imc_config, 
    #             w_q_int, None, 
    #             codebook_dict[config_name], 
    #             decomposer_dict[config_name], 
    #             faultmaps_dict[config_name])

    # decomposer = None
    final_rc, fawd_matched, stats = c_fault_free_solve(
            random_vector, codebook, faultmaps, decomposer, 
            imc_config['num_parallel_processes'])
    
    # compute sparsity of final_rc
    sparsity = np.count_nonzero(final_rc) / final_rc.size
    # print(sparsity)
    del stats['vec_in_range']
    del stats['vec_out_of_range']
    del stats['vec_remaining']
    del stats['vec_cont_representable']
    print_config(stats)
    stats['sparsity'] = 100 - (float(sparsity)*100)

    return stats

def inject_noise_single(layer_name, imc_config, w_q_int, scaling_factor, codebook, decomposer, faultmaps):
    # move w_q_int to cpu and make it into np array
    w_q_int_shape = w_q_int.shape
    num_elements = w_q_int.size
    w_q_int_np = w_q_int.reshape(-1,1).astype(np.int64)

    faultmaps.gen_fault_map(num_elements)
    final_rc, fawd_matched, stats = c_fault_free_solve(
        w_q_int_np, codebook, faultmaps, decomposer, 1)
    
    saf0 = faultmaps.map_saf0_list
    safa = faultmaps.map_all_faults_list
    vec = codebook.sig_vec
    w_q_int_faulty = final_rc*np.logical_not(safa) + saf0*(codebook.L-1)
    w_q_int_faulty[:,1] *= -1
    w_q_int_faulty = np.einsum('k,ijkl->i', vec, w_q_int_faulty).astype(np.int64)
    w_q_int_faulty = w_q_int_faulty.reshape(-1,1)

    # Todo: support for energy scalable code in the future
    # w_q_int_faulty = np.einsum('k,ijkl->il', vec, w_q_int_faulty).astype(np.int64)
    # w_q_int_faulty = np.cumsum(w_q_int_faulty, axis=1)
    # w_q_int_faulty = w_q_int_faulty[codebook.R_start:].reshape(-1,1)

    avg_residual = np.mean(np.abs(w_q_int_np - w_q_int_faulty))
    # zero_error = np.sum(w_q_int_np == w_q_int_faulty) / num_elements
    # print(f"{layer_name} - w_q_int_shape  = {w_q_int_shape}")
    # print(f"{layer_name} - num_elements   = {num_elements}")
    # print(f"{layer_name} - w_q_int_np     = {w_q_int_np[0:10]}")
    # print(f"{layer_name} - w_q_int_faulty = {w_q_int_faulty[0:10]}")
    # print(f"{layer_name} - avg residual   = {avg_residual}")
    # print(f"{layer_name} - zero error     = {zero_error}")

    w_q_int_faulty = w_q_int_faulty.reshape(w_q_int_shape)
    # w_q_faulty = w_q_int_faulty * scaling_factor

    return layer_name, w_q_faulty, avg_residual, stats

if __name__ == '__main__':
    set_start_method('fork')
    main()