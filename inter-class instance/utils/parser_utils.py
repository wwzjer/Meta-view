from torch import cuda

def str2bool(v):
    if v.lower() in ['y', 'yes', 't', 'true']:
        return True
    return False

def get_args():
    import argparse
    import os
    import torch
    import json
    parser = argparse.ArgumentParser(description='Welcome to the MAML++ training and inference system')

    parser.add_argument('--batch_size', nargs="?", type=int, default=2, help='Batch_size for experiment')
    parser.add_argument('--image_height', nargs="?", type=int, default=84)
    parser.add_argument('--image_width', nargs="?", type=int, default=84)
    parser.add_argument('--image_channels', nargs="?", type=int, default=1)
    parser.add_argument('--gpu_to_use', type=int,default=1)
    parser.add_argument('--num_dataprovider_workers', nargs="?", type=int, default=0)
    parser.add_argument('--max_models_to_save', nargs="?", type=int, default=5)

    parser.add_argument('--dataset_name', type=str, default="ModelNet_mat")
    parser.add_argument('--dataset_path', type=str, default="ModelNet_mat")
    parser.add_argument('--experiment_name', nargs="?", type=str, default="T3_N5_actor_mask")
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--val_seed', type=int, default=0)
    parser.add_argument('--sets_are_pre_split', type=str, default="True")
    parser.add_argument('--train_val_test_split', nargs='+', default=[0.64, 0.16, 0.20])    
    parser.add_argument('--evaluate_on_test_set_only', type=str, default="True")

    parser.add_argument('--total_epochs', type=int, default=200, help='Number of epochs per experiment')
    parser.add_argument('--total_iter_per_epoch', type=int, default=500, help='Number of iters per epoch')
    parser.add_argument('--continue_from_epoch', nargs="?", type=str, default='latest', help='Continue from checkpoint of epoch')
    parser.add_argument('--num_evaluation_tasks', type=int, default=200, help='Number of evaluation_tasks')
    parser.add_argument('--multi_step_loss_num_epochs', type=int, default=15, help='multi_step_loss_num_epochs')
    parser.add_argument('--learnable_per_layer_per_step_inner_loop_learning_rate', type=str, default="True")
    parser.add_argument('--enable_inner_loop_optimizable_bn_params', type=str, default="False")

    parser.add_argument('--max_pooling', type=str, default="True")
    parser.add_argument('--per_step_bn_statistics', type=str, default="True")
    parser.add_argument('--learnable_batch_norm_momentum', type=str, default="False")
    parser.add_argument('--load_into_memory', type=str, default="False")
    parser.add_argument('--init_inner_loop_learning_rate', type=float, default=0.01, help='init_inner_loop_learning_rate')
    parser.add_argument('--learnable_bn_gamma', type=str, default="True")
    parser.add_argument('--learnable_bn_beta', type=str, default="True")

    parser.add_argument('--dropout_rate_value', type=float, default=0., help='Dropout_rate_value')
    parser.add_argument('--min_learning_rate', type=float, default=0.001, help='Min learning rate')
    parser.add_argument('--meta_learning_rate', type=float, default=0.001, help='Learning rate of overall MAML system')
    parser.add_argument('--total_epochs_before_pause', type=int, default=101)
    parser.add_argument('--first_order_to_second_order_epoch', type=int, default=-1)

    parser.add_argument('--norm_layer', type=str, default="batch_norm")
    parser.add_argument('--embed_size', type=int,default=128)
    parser.add_argument('--kernel_size', type=int, default=5, help='Number of classes to sample per set')
    parser.add_argument('--cnn_num_filters', type=int, default=48, help='Number of classes to sample per set')
    parser.add_argument('--num_stages', type=int, default=4, help='Number of stages')
    parser.add_argument('--conv_padding', type=str, default="True")
    parser.add_argument('--number_of_training_steps_per_iter', type=int, default=1, help='Number of classes to sample per set')
    parser.add_argument('--number_of_evaluation_steps_per_iter', type=int, default=1, help='Number of classes to sample per set')
    parser.add_argument('--cnn_blocks_per_stage', type=int, default=1)
    parser.add_argument('--num_classes_per_set', type=int, default=5, help='Number of classes to sample per set')
    parser.add_argument('--num_samples_per_class', type=int, default=1, help='Number of samples per set to sample')
    parser.add_argument('--num_target_samples', type=int, default=15)

    parser.add_argument('--second_order', type=str, default="True")
    parser.add_argument('--use_multi_step_loss_optimization', type=str, default="False")

    parser.add_argument('--reset_stored_filepaths', type=str, default="False")
    parser.add_argument('--reverse_channels', type=str, default="False")
    parser.add_argument('--num_of_gpus', type=int, default=1)
    parser.add_argument('--indexes_of_folders_indicating_class', nargs='+', default=[-3, -2])
    parser.add_argument('--samples_per_iter', nargs="?", type=int, default=1)
    parser.add_argument('--labels_as_int', type=str, default="False")
    parser.add_argument('--seed', type=int, default=104)


    parser.add_argument('--task_learning_rate', type=float, default=0.01, help='Learning rate per task gradient step')

    # parser.add_argument('--name_of_args_json_file', type=str, default="experiment_config/ModelNet40.json")

###############################################Active Selection############################################################
###########################################################################################################################
    # Optimization options
    # parser.add_argument('--h5_path', type=str, default='./data/ModelNet/recognition_modelnet10.h5')
    # parser.add_argument('--epochs', type=int, default=400)
    # parser.add_argument('--lr', type=float, default=7e-3)
    # parser.add_argument('--final_lr', type=float, default=1e-5)
    # parser.add_argument('--saturate_epoch', type=int, default=800)
    # parser.add_argument('--weight_decay', type=float, default=5e-3)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--init', type=str, default='uniform', help='[ xavier | normal | uniform]')
    # parser.add_argument('--shuffle', type=str2bool, default=True)
    # parser.add_argument('--combineDropout', type=float, default=0)
    parser.add_argument('--lambda_reward', type=float, default=1., help='Coefficient of entropy term in loss')
    parser.add_argument('--lambda_entropy', type=float, default=0.005, help='Coefficient of entropy term in loss')
    # parser.add_argument('--lambda_la', type=float, default=1.5)
    # parser.add_argument('--featDropout', type=float, default=0)
    # parser.add_argument('--rnn_type', type=int, default=0, help='[ 0 - SimpleRNN | 1 - LSTM | 2 - Standard RNN]')
    # parser.add_argument('--normalize_hidden', default=True, help='Are hidden vectors normalized before classification?')
    # parser.add_argument('--nonlinearity', default='relu', help='[ relu | tanh ]')
    # parser.add_argument('--optimizer_type', default='sgd', help='[ adam | sgd ]')

    # Agent options
    parser.add_argument('--actOnElev', type=str2bool, default=False) # original true
    parser.add_argument('--actOnAzim', type=str2bool, default=False)
    parser.add_argument('--actOnTime', type=str2bool, default=False) # original true
    parser.add_argument('--load_model', type=str, default='')
    # parser.add_argument('--greedy', type=str2bool, default=False, help='enable greedy action selection during validation?')
    # parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--actorType', type=str, default='actor', help='[ actor | random | saved_trajectories | peek_saliency | const_action ]')
    # parser.add_argument('--baselineType', type=str, default='average', help='[ average ]')
    # parser.add_argument('--addExtraLinearFuse', type=str2bool, default=False)
    # parser.add_argument('--trajectories_type', type=str, default='utility_maps', help='[ utility_maps | expert_trajectories ]')
    # parser.add_argument('--utility_h5_path', type=str, default='', help='Stored utility maps from one-view expert to obtain expert trajectories')

    # Environment options
    parser.add_argument('--T', type=int, default=3, help='Number of allowed steps / views')
    parser.add_argument('--M', type=int, default=6, help='Number of azimuths')
    parser.add_argument('--N', type=int, default=5, help='Number of elevations')
    parser.add_argument('--delta_M', type=int, default=5, help='azim neighborhood for actions')
    parser.add_argument('--delta_N', type=int, default=5, help='elev neighborhood for actions')
    # parser.add_argument('--F', type=int, default=1200, help='Image feature size')
    # parser.add_argument('--rewards_greedy', type=str2bool, default=False, help='enable greedy rewards?')
    parser.add_argument('--wrap_azimuth', type=str2bool, default=True, help='wrap around the azimuths when rotating?')
    parser.add_argument('--wrap_elevation', type=str2bool, default=False, help='wrap around the elevations when rotating?')
    parser.add_argument('--reward_scale', type=float, default=1, help='Scaling overall reward during REINFORCE')

    # Evaluation options
    # parser.add_argument('--model_path', type=str, default='model_best.net')
    # parser.add_argument('--eval_val', type=str2bool, default=False, help='evaluate on validation split?')
    # parser.add_argument('--compute_all_times', type=str2bool, default=False, help='evaluate model at all time steps?')
    # parser.add_argument('--average_over_time', type=str2bool, default=False, help='Average classifier activations at each time step?')
    ###########################################################################################################################


    args = parser.parse_args()
    args_dict = vars(args)
    # if args.name_of_args_json_file is not "None":
    #     args_dict = extract_args_from_json(args.name_of_args_json_file, args_dict)

    for key in list(args_dict.keys()):

        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False
        if key == "dataset_path":
            args_dict[key] = os.path.join(os.environ['DATASET_DIR'], args_dict[key])
            print(key, os.path.join(os.environ['DATASET_DIR'], args_dict[key]))

        print(key, args_dict[key], type(args_dict[key]))

    args = Bunch(args_dict)


    args.use_cuda = torch.cuda.is_available()

    if args.gpu_to_use == -1:
        args.use_cuda = False

    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_to_use)
        device = cuda.current_device()
    else:
        device = torch.device('cpu')

    return args, device



class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def extract_args_from_json(json_file_path, args_dict):
    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        if "continue_from" in key:
            pass
        elif "gpu_to_use" in key:
            pass
        else:
            args_dict[key] = summary_dict[key]

    return args_dict





