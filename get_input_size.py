from utils import exp_utils
from utils.utils import save_dict
from utils.exp_utils import setup_exp
from utils.nn_utils import get_input_size_with_hooks
from os.path import exists, join

json_path = "./utils/act_dim_data"

def main():
    list_of_models = [
        'vgg16q',
        'mobilenetv2q',
        'resnet50q',
        'resnet18q',
        'resnet20q',
    ]
    for idx, model_name in enumerate(list_of_models):
        # check if the json file already exists
        if exists(join(json_path, f"{model_name}.json")):
            print(f"[{idx+1}/{len(list_of_models)}] Skipping {model_name}")
            continue

        print(f"[{idx+1}/{len(list_of_models)}] Calculating for {model_name}")
        calc_and_save_act_dim_as_json(model_name)

def calc_and_save_act_dim_as_json(model_name): 
    nn_config = exp_utils.base_nn_config.copy()
    nn_config['model'] = model_name
    nn_config['num_workers'] = 16
    nn_config['lead_gpu_idx'] = 3
    nn_config['all_gpu_idx'] = [3]
    nn_config['batch_size'] = 64
    nn_config['half_prec'] = False
    nn_config['acti_bit'] = [32]
    nn_config['weight_bit'] = [8, 32]

    model, load_train, load_val, f_eval, f_adap = setup_exp(nn_config)

    _, train_loader = load_train()

    input_size_dict = get_input_size_with_hooks(
        model, next(iter(train_loader))[0].cuda())

    save_dict(input_size_dict, join(json_path, f"{model_name}.json"))

if __name__ == "__main__":
    main()