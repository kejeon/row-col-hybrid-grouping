import torch.nn as nn
import yaml
from itertools import product
from functools import partial

from .utils import setup_dir, logger_info, logger_newline, logger_bold, logger_lvl1, logger_lvl2, logger_lvl3, green
from .nn_utils import setup_gpus, get_dataset_for_model, get_model, forward_gen_func
from rc_grouping.rccodes import calc_q_lvl_RC


# logger = None

R1C8 = {
    'name': 'R1C4', 'r_start': 0,
    'num_row_grouping': 1, 'num_col_grouping': 8}
R2C8 = {
    'name': 'R2C2', 'r_start': 1,
    'num_row_grouping': 2, 'num_col_grouping': 8}
R2C8 = {
    'name': 'R2C4', 'r_start': 1,
    'num_row_grouping': 2, 'num_col_grouping': 8}

R1C4 = {
    'name': 'R1C4', 'r_start': 0,
    'num_row_grouping': 1, 'num_col_grouping': 4}
R2C2 = {
    'name': 'R2C2', 'r_start': 1,
    'num_row_grouping': 2, 'num_col_grouping': 2}
R2C4 = {
    'name': 'R2C4', 'r_start': 1,
    'num_row_grouping': 2, 'num_col_grouping': 4}

R1C2 = {
    'name': 'R1C2', 'r_start': 0,
    'num_row_grouping': 1, 'num_col_grouping': 2}
R2C2 = {
    'name': 'R2C2', 'r_start': 1,
    'num_row_grouping': 2, 'num_col_grouping': 2}
R2C1 = {
    'name': 'R2C1', 'r_start': 1,
    'num_row_grouping': 2, 'num_col_grouping': 1}

base_nn_config = {
    'lead_gpu_idx': 0,
    'all_gpu_idx': [0],
    'acti_bit': [4],
    'weight_bit': [1, 2, 4, 8],
    'batch_size': 256,
    'num_workers': 16,
    'model': 'resnet50q',
    'half_prec': True,
    
}

base_imc_config = {
    'imc_array_dim': 512,
    'rc_config': None,
    'rc_config_prec': None,
    'cell_precision': 2,
    'shift_base': 2,
    'p_fault': None,
    'p_saf0': 0.0175,
    'p_saf1': 0.0904,
    'num_parallel_processes': 20
}

iter_imc_config_4col_simple = {
    'rc_config': [
        [R1C4],
        [R2C2]
    ]
}

iter_imc_config_8col_simple = {
    'rc_config': [
        [R1C8],
        [R2C4],
        [R2C8]
    ]
}

iter_imc_config_4col_mixed = {
    'rc_config': [
        [R1C4, R2C2], #throughput focused
        [R1C4, R2C4], #accuracy focused
        [R1C4, R2C2, R2C4]  #balanced
    ]
}

iter_imc_config_8col_mixed = {
    'rc_config': [
        [R1C8, R2C4], #throughput focused
        [R1C8, R2C8], #accuracy focused
        [R1C8, R2C4, R2C8]  #balanced
    ]
}
    

def iterate_all_configs(fixed_config, iter_config):
    all_configs = []
    iter_combinations = list(product(*iter_config.values()))
    for combo in iter_combinations:
        config = fixed_config.copy()
        for key, value in zip(iter_config.keys(), combo):
            config[key] = value
        all_configs.append(config)
    return all_configs

def elaborate_configs(all_configs):
    for config in all_configs:
        # fix shift_base = cell_precision
        config['shift_base'] = config['cell_precision']

        # fix saf0 and saf1 ratio
        if config['p_fault'] is None:
            config['p_saf0'] = 0.0175
            config['p_saf1'] = 0.0904
        else:
            p_ratio_total = 0.0175 + 0.0904
            config['p_saf0'] = 0.0175/p_ratio_total * config['p_fault']
            config['p_saf1'] = 0.0904/p_ratio_total * config['p_fault']

        # compute rc_config_prec based on rc_config
        rc_config_prec = []
        for rc in config['rc_config']:
            prec = calc_q_lvl_RC(
                R=rc['num_row_grouping'], 
                C=rc['num_col_grouping'],
                mem_q_lvl=config['cell_precision'],
                shift_base=config['shift_base'])
            rc_config_prec.append(int(prec))
        config['rc_config_prec'] = rc_config_prec

    return all_configs



def print_config(config):
    for key, value in config.items():
        if key is not 'rc_config':
            logger_lvl1(f"- {key}: {green(value)}")
            continue

        if type(value) is not list:
            logger_lvl1(f"- {key}: {green([rc['name'] for rc in value])}")
            continue

        if type(value) is list:
            logger_lvl1(f"- {key}:")

            for idx, my_list in enumerate(value):
                if type(my_list) is list:
                    logger_lvl2(f"- {key} combo {idx+1}:")
                    for rc in my_list:
                        logger_lvl3(f"- {rc['name']}")
                else:
                    logger_lvl2(f"- {green(my_list['name'])}")
            continue

        raise ValueError("Invalid config type")
    return

def setup_exp(nn_config):
    global forward_eval, forward_adap
    logger_bold(f"=====================================")
    logger_bold(f"====== EXPERIMENT SETUP SCRIPT ======")
    logger_bold(f"=====================================")

    logger_newline()
    logger_bold(f"Setting up GPUs...")
    num_gpus = len(nn_config['all_gpu_idx'])
    device = setup_gpus(nn_config['lead_gpu_idx'])
    temp_str = green(f"cuda:{nn_config['lead_gpu_idx']}")
    logger_lvl1(f"- Number of GPUs: {green(num_gpus)}")
    logger_lvl1(f"- Lead GPU index: {temp_str}")
    if num_gpus > 1:
        logger_lvl1(f"- All GPU indices: {green(nn_config['all_gpu_idx'])}")

    logger_newline()
    logger_bold(f"Loading dataset...")

    load_train = partial(
        get_dataset_for_model, 
        model_name=nn_config['model'],
        split='train', 
        batch_size=nn_config['batch_size'], 
        num_workers=nn_config['num_workers'])
    load_val = partial(
        get_dataset_for_model, 
        model_name=nn_config['model'],
        split='val', 
        batch_size=nn_config['batch_size'], 
        num_workers=nn_config['num_workers'])

    val_data, val_loader = load_val()
    # train_data, train_loader = get_dataset_for_model(
    #     model_name=nn_config['model'],
    #     split='train',
    #     batch_size=nn_config['batch_size'],
    #     num_workers=nn_config['num_workers']
    # )
    # val_data, val_loader = get_dataset_for_model(
    #     model_name=nn_config['model'],
    #     split='val',
    #     batch_size=nn_config['batch_size'],
    #     num_workers=nn_config['num_workers']
    # )
    logger_lvl1(f"- Dataset load function generated")

    logger_newline()
    logger_bold(f"Creating model...")
    model = get_model(
        model_name=nn_config['model'],
        acti_bit=nn_config['acti_bit'],
        weight_bit=nn_config['weight_bit'],
        num_classes=val_data.num_classes,
        all_gpu_idx=nn_config['all_gpu_idx'],
        device=device
    )

    if nn_config['half_prec']:
        model = model.half()
    criterion = nn.CrossEntropyLoss().to(device)

    forward_functions = forward_gen_func(
        device=device, 
        criterion=criterion, 
        half_prec=nn_config['half_prec'])
    forward_eval = forward_functions['forward_evaluation']
    forward_adap = forward_functions['forward_adaptiveBN']

    logger_lvl1(f"Model created: {green(nn_config['model'])}")

    return model, load_train, load_val, forward_eval, forward_adap

