# Row Column Hybrid Grouping

Welcome to the repository of the paper "Row-Column Hybrid Grouping for Fault-Resilient
Multi-Bit Weight Representation on IMC Arrays". 

This repository contains full implementation of the proposed method and its constituent components:

1. Our implementation of [Fault-Free](https://ieeexplore.ieee.org/abstract/document/9976251) algorithm accelerated with [Numba](https://numba.pydata.org/)
2. Proposed encoding/decoding algorithms of integer value into row-column hybrid grouping representation
3. Proposed Integer Linear Programming (ILP) Fault-Free using [Gurobi](https://www.gurobi.com/) solver
4. Proposed compiler pipeline
5. Pytorch-based simulation framework for stuck-at-faults (SAF)

## Getting Started
### Environment
* Python 3.7.13
* PyTorch 1.12.1 
* CUDA 11.4

We used the following docker image: [pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel](https://hub.docker.com/layers/pytorch/pytorch/1.12.1-cuda11.3-cudnn8-devel/images/sha256-dda4e7ce91e3f5b309233111b251e54cf47b44a742fe37c7f68d9429321fa0f9?context=explore)

### Dependencies
* numpy
* pandas
* [numba](https://numba.pydata.org/)
* [gurobipy](https://pypi.org/project/gurobipy/)
* [tqdm](https://tqdm.github.io/)
* [tensorboardX](https://github.com/lanpa/tensorboardX)
* [gpustat](https://github.com/wookayin/gpustat)

Run the command below to install all dependencies. 
```
pip install -r requirements.txt
```
### Obtain Gurobi Academic License
To use the Gurobi Solver, you need to obtain a license. Fortunately, Gurobi offers free licenses for academic users. Follow the instructions provided [here](https://www.gurobi.com/features/academic-wls-license/) to obtain your license.

### Download Prerained Models
Below are the pretrained models that we used in our experiments. 
* [ResNet-20](https://drive.google.com/file/d/1-17bNioG0Dd9oN6uz-EQy1RBRSxUkzMz/view?usp=sharing) (CIFAR10)
* [ResNet-18](https://drive.google.com/file/d/1-1pBLRrDc63ff7HmYmoJSQsI2QnR2yg5/view?usp=sharing) (ImageNet)
* [ResNet-50](https://drive.google.com/file/d/1-A304sGNfa_oTt4wsMcSrm11J1UFPwOO/view?usp=sharing) (ImageNet)
* [VGG16-bn]() (ImageNet)

### Running the code
1. Place your pre-trained models in the `_pretrained/` directory (you can download them from the "Pretrained models" section below)
2. Modify the code in the `exp_noise_inject.py` file to configure the experiment
3. Run `exp_noise_inject.py`

## Repo Structure

* `_pretrained/`: Put your trained models here. You can download the models from the "Pretrained models" section below
* `exp_results/`: Contains the logs and results for experiments. 
* `dnn_model/`: Contains the code for running our deep learning models; adapted from [AnyPrecision](https://github.com/SHI-Labs/Any-Precision-DNNs) repo. 
* `rc_grouping/`: Contains the implemenation of encoding/decoding code for row-column hybrid grouping
* `fault_free/`: Contains the implemenation of the proposed compiler pipeline
