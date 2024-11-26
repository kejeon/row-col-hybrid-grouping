from .utils import load_dict

dataset_model_dict = {
    'vgg16q': 'imagenet',
    'mobilenetv2q': 'imagenet',
    'resnet50q': 'imagenet',
    'resnet18q': 'imagenet',
    'resnet20q': 'cifar10',
    'resnet20qe1': 'cifar10',
}

pretrained_paths = {
    'vgg16q': './_pretrained/vgg16bn_epoch35.pth.tar',
    'mobilenetv2q': './_pretrained/mobile_net_best.pth.tar',
    'resnet50q'  : './_pretrained/resnet50q_any_recursive.pth.tar',
    'resnet18q'  : './_pretrained/RN18Q_1248.pth.tar',
    'resnet20q'  : './_pretrained/RN20Q_1248_3567.pth.tar',
    'resnet20qe1': './_pretrained/RN20Q_124832_exp1.pth.tar',
}

parallel_settings = {
    'vgg16q': False,
    'mobilenetv2q': True,
    'resnet50q': True,
    'resnet18q': False,
    'resnet20q': False,
    'resnet20qe1': False,
}

act_dim_dict = {}
def load_act_dim_data():
    global act_dim_dict
    list_of_models = list(dataset_model_dict.keys())
    for model in list_of_models:
        model_dict = load_dict(f"./utils/act_dim_data/{model}.json")
        act_dim_dict[model] = model_dict
    return act_dim_dict