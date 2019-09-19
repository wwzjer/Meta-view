# Meta-view

## Requirements
    Python 3.7.3
    CUDA == 9.2
    PyTorch == 1.1.0
    tqdm == 4.31.1
    numpy == 1.16.2
    tersorboardX == 1.8
    scipy == 1.2.1
  
## Dataset
You can find the rendered viewgrid dataset ModelNet-40 in .mat format here (https://drive.google.com/file/d/1xoYwJpiDlJcro-4vkZ_l3ZSfbidsIy7k/view?usp=sharing). Download it to the datasets folder.

    cd inter-class instance
    mkdir datasets
    unzip ModelNet_mat
    
 ## Run
 ### For inter-class instance recongition experiment
 #### Training

    cd inter-class instance
    CUDA_VISIBLE_DEVICES=0 python train_maml_system.py
#### Testing
    CUDA_VISIBLE_DEVICES=0 python train_maml_system.py --evalute_on_test_set_only True
