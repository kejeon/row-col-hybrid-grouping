import torch
import torch.nn as nn
import torch.optim

import gc
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
# from c_fault_free.utils.test_utils import compute_faulty_q_code, compute_residual_multiple

from utils.nn_utils import get_layer_by_name, set_precision
from utils.nn_utils import quantize_conv2d, q_freeze_conv2d, q_unfreeze_conv2d
from utils.exp_utils import print_config, setup_exp, iterate_all_configs, elaborate_configs
from utils.exp_utils import *
from utils.nn_data import act_dim_dict, load_act_dim_data
import utils.exp_utils as exp_utils

from utils.utils import set_timezone, setup_log_dir, create_logger, save_dict, load_dict
from utils.utils import setup_dir, logger_info, logger_newline, logger_bold, logger_lvl1, logger_lvl2, green

from c_fault_free.c_fault_free import c_fault_free_adv_solve as c_fault_free_solve
# from c_fault_free.c_fault_free import c_fault_free_solve


# To prevent experiments from crashing due to truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

forward_eval = None
forward_adap = None
# exp_result_path = None

def main(my_p_fault, my_model, exp_base_path=None, skip_layer=None, 
         train_loader=None, val_loader=None, pass_loaders=False):
    global forward_eval, forward_adap
    gc.disable()

    # set exp configurations
    NUM_TRIAL = 10
    nn_config = exp_utils.base_nn_config.copy()
    nn_config['model'] = my_model
    nn_config['num_workers'] = 32
    nn_config['lead_gpu_idx'] = 2
    nn_config['all_gpu_idx'] = [nn_config['lead_gpu_idx']]
    nn_config['batch_size'] = 512
    nn_config['acti_bit'] = [8]
    # nn_config['weight_bit'] = [2,8]
    nn_config['weight_bit'] = [4,8]
    nn_config['half_prec'] = True
    # nn_config = load_dict('./exp_results/resnet20qe1__2024-09-11_17-48-10/nn_config.yaml')
    act_dim_dict = load_act_dim_data()
    act_dim_dict = act_dim_dict[nn_config['model']]

    imc_config = exp_utils.base_imc_config.copy()
    imc_config['num_parallel_processes'] = 4
    imc_config['cell_precision'] = 4
    imc_config['imc_array_dim'] = 64
    imc_config['p_fault'] = my_p_fault
    iter_configs = iter_imc_config_4col_simple.copy()
    iter_configs = {
        'rc_config': [
            [R1C4],
            # [R2C1],
            # [R2C4],
            [R2C2],
            # [R1C4, R2C4],
            # [R1C4, R2C2, R2C4],
        ]
    }
    imc_config_list = iterate_all_configs(imc_config, iter_configs)
    imc_config_list = elaborate_configs(imc_config_list)

    if exp_base_path is None:
        exp_result_path = f"./exp_results/{nn_config['model']}_"
    else:
        exp_result_path = f"./exp_results/{exp_base_path}/{nn_config['model']}_"
    exp_result_path = setup_dir(exp_result_path)
    
    set_timezone()
    logger = create_logger('exp_noise_inject', exp_result_path)
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
    if pass_loaders is True and train_loader is None and val_loader is None:
        _, val_loader = load_val()
        _, train_loader = load_train()
        load_val = None
        load_train = None
    elif pass_loaders is False:
        val_loader = None
        train_loader = None

    logger_newline()
    logger_bold(f"=====================================")
    logger_bold(f"====== NOISE INJECTION SCRIPT =======")
    logger_bold(f"=====================================")
    for idx, imc_config in enumerate(imc_config_list):
        logger_bold(f"Using imc_config {idx+1}/{len(imc_config_list)}")
        run_path = f"{exp_result_path}/run_{idx}"
        run_path = setup_dir(run_path)
        final_result = run_exp(
            imc_config, model, act_dim_dict=act_dim_dict, exp_idx=idx, 
            num_trial=NUM_TRIAL, ada_bn_repeat=1,
            load_val=load_val, load_train=load_train, 
            val_loader=val_loader, train_loader=train_loader, 
            run_path=run_path, skip_layer=skip_layer)
        # logger_lvl1(f"Top1 mean: {top1_mean:.2f}%, Top1 std: {top1_std:.2f}%")
        save_dict(final_result, f"{run_path}/final_result.yaml")
        # save_dict(compile_stats_mean, f"{run_path}/compile_stats_mean.yaml")
        # save_dict(compile_stats_total, f"{run_path}/compile_stats_total.yaml")

    return val_loader, train_loader

def run_exp(imc_config, model, act_dim_dict, num_trial, load_val, val_loader, 
            load_train=None, train_loader=None, ada_bn_repeat=1, exp_idx=0, run_path=None, skip_layer=None):
    pass_loaders = load_val is None
    # train_loader = None
    # if run_path is None:
    #     run_path = setup_dir(f"{exp_result_path}/run_{exp_idx}")

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

    # Evaluate the model
    logger_newline( )
    logger_bold(f"Evaluating the model...")
    logger_lvl1(f"Loading val dataset...")
    if val_loader is None:
        _, val_loader = load_val()
    top1_b, top5_b, loss = forward_eval(model=model, dataset_loader=val_loader)
    logger_lvl1(f"Evaluation done: Top1: {top1_b:.2f}%, Top5: {top5_b:.2f}%, Loss: {loss:.2f}")

    # Run adaptive BN if train_loader is not None
    if load_train is not None or train_loader is not None:
        logger_newline()
        logger_bold(f"Running adaptive BN...")
        logger_lvl1(f"Loading train dataset...")
        if train_loader is None:
            _, train_loader = load_train()
        for i in range(ada_bn_repeat):
            logger_lvl1(f"Iteration {i+1}/{ada_bn_repeat}...")
            _, _, _ = forward_adap(model=model, dataset_loader=train_loader)

        top1_a, top5_a, loss = forward_eval(model=model, dataset_loader=val_loader)
        logger_lvl2(f"AdaBN iteration done: Top1: {top1_a:.2f}%, Top5: {top5_a:.2f}%, Loss: {loss:.2f}")
        # logger_lvl1(f"Trashing train/val dataset...")
        # train_loader = None
        # val_loader = None
        # gc.collect()
    gc.freeze()
    gc.enable()
    if pass_loaders is False:
        del train_loader
    gc.collect()

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
        if num_row_grouping == 2 and num_col_grouping == 4:
            decomposer = None
        else:
            decomposer = Decomp(rc_codebook=codebook) 
            decomposer.decomp_dict_rc = None
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

    top1_list = []
    top5_list = []
    compile_stats_mean = None
    logger_newline()
    logger_bold(f"Evaluating faulty model...")
    acc_df = None
    for i in range(num_trial):
        model, stats = inject_noise(imc_config, layer_config, model, codebook_dict, decomposer_dict, faultmaps_dict, skip_layer=skip_layer)
        if acc_df is None:
            acc_df = pd.DataFrame(stats)
        else:
            temp_df = pd.DataFrame(stats)
            acc_df = acc_df.add(temp_df, fill_value=0)
        top1, top5, loss = forward_eval(model=model, dataset_loader=val_loader)
        logger_lvl1(f"Trial {i+1}/{num_trial} done: Top1: {top1:.2f}%, Top5: {top5:.2f}%, Loss: {loss:.2f}")
        top1_list.append(top1)
        top5_list.append(top5)

    acc_df = acc_df/num_trial
    # store as csv
    acc_df.to_csv(f"{run_path}/avg_per_layer_stats.csv")

        # if train_loader is not None:
        #     logger_newline()
        #     logger_bold(f"Running adaptive BN...")
        #     for i in range(ada_bn_repeat):
        #         logger_lvl1(f"Iteration {i+1}/{ada_bn_repeat}...")
        #         _, _, _ = forward_adap(model=model, dataset_loader=train_loader)

        #     top1_a, top5_a, loss = forward_eval(model=model, dataset_loader=val_loader)
        #     logger_lvl2(f"AdaBN iteration done: Top1: {top1_a:.2f}%, Top5: {top5_a:.2f}%, Loss: {loss:.2f}")

        # get the mean and std of compile stats
    #     if compile_stats_mean is None:
    #         compile_stats_mean = compile_stats
    #     else:
    #         for layer_name in compile_stats_mean.keys():
    #             # compile_stats_mean[key] += compile_stats[key]
    #             for key in compile_stats_mean[layer_name].keys():
    #                 compile_stats_mean[layer_name][key] += compile_stats[layer_name][key]


    # # for layer_name in compile_stats_mean.keys():
    # #     for key in compile_stats_mean[layer_name].keys():
    # #         compile_stats_mean[layer_name][key] /= num_trial

    # # compile_stats_total = {}
    # # for layer_name in compile_stats_mean.keys():
    # #     for key in compile_stats_mean[layer_name].keys():
    # #         if key not in compile_stats_total:
    # #             compile_stats_total[key] = compile_stats_mean[layer_name][key]
    # #         else:
    # #             compile_stats_total[key] += compile_stats[layer_name][key]

    top1_mean = np.mean(top1_list)
    top1_std = np.std(top1_list)
    top5_mean = np.mean(top5_list)
    top5_std = np.std(top5_list)

    final_result = {
        'top1_mean': top1_mean,
        'top1_std': top1_std,
        'top5_mean': top5_mean,
        'top5_std': top5_std,
        'top1_before_abn': top1_b,
        'top5_before_abn': top5_b,
        'top1_after_abn': top1_a,
        'top5_after_abn': top5_a,
        'total_cycle': total_cycle,
    }

    # convert final result into float
    for key in final_result.keys():
        final_result[key] = float(final_result[key])

    return final_result

def inject_noise(imc_config, layer_config, model, codebook_dict, decomposer_dict, faultmaps_dict, skip_layer=None):
    total_stats = {}
    futures = []
    avg_residual_list = []
    print(skip_layer)

    tic = time()
    with ProcessPoolExecutor(max_workers=64) as executor:
        for layer_name, rc_config in reversed(layer_config):
            layer = get_layer_by_name(model, layer_name)
            w_q_int = layer.w_q_int.detach().cpu().numpy()
            scaling_factor = (layer.scaling_factor).detach().cpu().numpy()
            config_name = rc_config['name']
            logger_lvl1(f"Submitted {layer_name}...")

            if skip_layer is not None and layer_name in skip_layer:
                logger_lvl1(f"Skipping {layer_name}...")
                continue

            result = executor.submit(
                inject_noise_single, 
                layer_name, imc_config, 
                w_q_int, scaling_factor, 
                codebook_dict[config_name], 
                decomposer_dict[config_name], 
                faultmaps_dict[config_name])
            futures.append(result)
        for future in futures:
            layer_name, w_q_faulty, avg_residual, stats = future.result()
            logger_lvl1(f"Injecting {layer_name}...")
            layer = get_layer_by_name(model, layer_name)
            layer_dtype = layer.weight.dtype
            layer.w_q = torch.tensor(w_q_faulty, dtype=layer_dtype).cuda()

            quant_err = torch.sum(torch.abs(layer.w_q_int.detach()*layer.scaling_factor - layer.weight.detach())/w_q_faulty.size)
            fault_err = torch.sum(torch.abs(layer.w_q.detach() - layer.weight.detach())/w_q_faulty.size)
            stats['quant_err'] = quant_err.item()
            stats['fault_err'] = fault_err.item()
            total_stats[layer_name] = stats
            avg_residual_list.append(avg_residual)
            logger_newline()
            logger_lvl1(f"===== {layer_name} =====")
            print_config(stats)
    toc = time()
    elapsed_time = toc - tic
    # logger_newline()
    # logger_lvl1(avg_residual_list)
    logger_newline()
    logger_lvl1(f"Noise Injection Time: {elapsed_time:.2f}s")
    logger_newline()
    return model, total_stats

def inject_noise_single(layer_name, imc_config, w_q_int, scaling_factor, codebook, decomposer, faultmaps):
    # move w_q_int to cpu and make it into np array
    w_q_int_shape = w_q_int.shape
    num_elements = w_q_int.size
    w_q_int_np = w_q_int.reshape(-1,1).astype(np.int64)
    bypass_fawd = False

    # if decomposer is none bypass fawd
    if decomposer is None:
        bypass_fawd = True

    faultmaps.gen_fault_map(num_elements)
    final_rc, fawd_matched, stats = c_fault_free_solve(
        w_q_int_np, codebook, faultmaps, decomposer, 4, bypass_fawd=bypass_fawd)
    
    saf0 = faultmaps.map_saf0_list
    safa = faultmaps.map_all_faults_list
    vec = codebook.sig_vec
    w_q_int_faulty = final_rc*np.logical_not(safa) + saf0*(codebook.L-1)
    w_q_int_faulty[:,1] *= -1
    w_q_int_faulty = np.einsum('k,ijkl->i', vec, w_q_int_faulty).astype(np.int64)
    w_q_int_faulty = w_q_int_faulty.reshape(-1,1)
    if bypass_fawd:
        vec_cont_rep = stats['vec_cont_representable']
        w_q_int_faulty[vec_cont_rep] = w_q_int_np[vec_cont_rep] 

    num_fawdable = np.sum(w_q_int_faulty == w_q_int_np)
    num_cvm = num_elements - num_fawdable
    num_cvm_in_remaining = num_cvm - stats['num_out_of_range']
    num_fawd_in_remaining = stats['num_remaining'] - num_cvm_in_remaining
    perc_fawdable = num_fawdable/num_elements
    perc_cvm = num_cvm/num_elements

    vec_out_of_range = stats['vec_out_of_range']
    err_out_of_range = np.sum(np.abs(w_q_int_np[vec_out_of_range] - w_q_int_faulty[vec_out_of_range]))
    err_total = np.sum(np.abs(w_q_int_np - w_q_int_faulty))
    err_reduced_precision = err_total - err_out_of_range

    stats['num_fawdable'] = num_fawdable
    stats['num_cvm'] = num_cvm
    stats['num_cvm_in_remaining'] = num_cvm_in_remaining
    stats['num_fawd_in_remaining'] = num_fawd_in_remaining
    stats['perc_fawdable'] = perc_fawdable
    stats['perc_cvm'] = perc_cvm
    stats['err_out_of_range'] = err_out_of_range
    stats['err_reduced_precision'] = err_reduced_precision
    stats['err_total'] = err_total

    del stats['vec_in_range']
    del stats['vec_out_of_range']
    del stats['vec_remaining']
    del stats['vec_cont_representable']

    # Todo: support for energy scalable code in the future
    # w_q_int_faulty = np.einsum('k,ijkl->il', vec, w_q_int_faulty).astype(np.int64)
    # w_q_int_faulty = np.cumsum(w_q_int_faulty, axis=1)
    # w_q_int_faulty = w_q_int_faulty[codebook.R_start:].reshape(-1,1)

    # avg_residual = np.mean(np.abs(w_q_int_np - w_q_int_faulty))

    w_q_int_faulty = w_q_int_faulty.reshape(w_q_int_shape)
    w_q_faulty = w_q_int_faulty * scaling_factor

    return layer_name, w_q_faulty, err_total, stats


# def inject_noise_no_parallel(imc_config, layer_config, model, codebook_dict, decomposer_dict, faultmaps_dict):
#     total_stats = {}
#     tqdm_layer_config = tqdm(layer_config)
#     for layer_name, rc_config in tqdm_layer_config:
#     # for layer_name, rc_config in layer_config:
#         tqdm_layer_config.set_description(f"Processing {layer_name}")
#         config_name = rc_config['name']
#         codebook = codebook_dict[config_name]
#         decomposer = decomposer_dict[config_name]
#         faultmaps = faultmaps_dict[config_name]

#         layer = get_layer_by_name(model, layer_name)
#         w_q_int = layer.w_q_int
#         scaling_factor = (layer.scaling_factor).detach().cpu().numpy()
#         # get number of elements in w_q_int
#         num_elements = w_q_int.numel()
#         w_q_int_shape = w_q_int.shape

#         # move w_q_int to cpu and make it into np array
#         w_q_int_np = w_q_int.detach().cpu().numpy()
#         w_q_int_np = w_q_int_np.reshape(-1,1).astype(np.int64)

#         # decompose 
#         faultmaps.gen_fault_map(num_elements)
#         tic = time()
#         final_rc, fawd_matched, stats = c_fault_free_solve(
#             w_q_int_np, codebook, faultmaps, decomposer, 
#             imc_config['num_parallel_processes'])
#         toc = time()
#         elapsed_time = toc - tic

#         total_stats[layer_name] = stats

#         # convert the rc code back to q_code
#         w_q_int_faulty = np.zeros_like(w_q_int_np)
#         for i in range(num_elements):
#             rc_code = final_rc[i]
#             all_fualts = faultmaps.map_all_faults_list[i]
#             saf0 = faultmaps.map_saf0_list[i]
#             w_q_int_faulty[i] = compute_faulty_q_code(
#                 rc_code, all_fualts, saf0, codebook)
            
#         # compute the residual
#         avg_residual = np.mean(np.abs(w_q_int_np - w_q_int_faulty))

#         # inject the faulty q code back to the layer
#         w_q_int_faulty = w_q_int_faulty.reshape(w_q_int_shape)
#         w_q_faulty = w_q_int_faulty * scaling_factor
#         w_q_faulty = torch.tensor(w_q_faulty, dtype=torch.float16).cuda()
#         layer.w_q = w_q_faulty

#         tqdm_layer_config.set_postfix(
#             elapsed_time=elapsed_time, avg_residual=avg_residual)

#     return model, total_stats

if __name__ == '__main__':
    set_start_method('spawn')
    # my_p_fault_list = np.arange(0.1,0.6,0.1)
    my_p_fault_list = [None]
    my_model_list = [
        'resnet20qe1',
        # 'resnet18q', 
        # 'resnet50q',
        # 'vgg16q',
    ]
    train_loader = None
    val_loader = None

    print(my_p_fault_list)
    print(my_model_list)
    
    for my_model in my_model_list:
        # exp_base_path = f"quant_vs_fault"
        exp_base_path = f"vary_fault_rate2/{my_model}"
        for my_p_fault in my_p_fault_list:
            if my_p_fault is not None:
                my_p_fault = float(my_p_fault)

            val_loader, train_loader = main(my_p_fault=my_p_fault, my_model=my_model, 
                                            exp_base_path=exp_base_path, skip_layer=None, pass_loaders=False, train_loader=train_loader, val_loader=val_loader)
        