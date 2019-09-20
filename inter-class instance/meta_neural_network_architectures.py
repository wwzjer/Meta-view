import numbers
from copy import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from visualize import visualize

def extract_top_level_dict(current_dict):
    """
    Builds a graph dictionary from the passed depth_keys, value pair. Useful for dynamically passing external params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param value: Param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return: A dictionary graph of the params already added to the graph.
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("layer_dict.", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    #print(current_dict.keys(), output_dict.keys())
    return output_dict


class MetaConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, dilation_rate=1):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting.
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Convolutional kernel size
        :param stride: Convolutional stride
        :param padding: Convolution padding
        :param use_bias: Boolean indicating whether to use a bias or not.
        """
        super(MetaConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.groups = int(groups)
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: Input image batch.
        :param params: If none, then conv layer will use the stored self.weights and self.bias, if they are not none
        then the conv layer will use the passed params as its parameters.
        :return: The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
        else:
            #print("No inner loop params")
            weight, bias = self.weight, self.bias

        out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)
        return out


class MetaLinearLayer(nn.Module):
    def __init__(self, input_shape, num_filters):
        """
        A MetaLinear layer. Applies the same functionality of a standard linearlayer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the linear layer. Useful for inner loop optimization in the meta
        learning setting.
        :param input_shape: The shape of the input data, in the form (b, f)
        :param num_filters: Number of output filters
        :param use_bias: Whether to use biases or not.
        """
        super(MetaLinearLayer, self).__init__()
        b, c = input_shape

        self.weights = nn.Parameter(torch.ones(num_filters, c))
        nn.init.xavier_uniform_(self.weights)
        self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        """
        Forward propagates by applying a linear function (Wx + b). If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param x: Input data batch, in the form (b, f)
        :param params: A dictionary containing 'weights' and 'bias'. If params are none then internal params are used.
        Otherwise the external are used.
        :return: The result of the linear function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weights"], params["bias"]
        else:
            pass
            #print('no inner loop params', self)

            weight, bias = self.weights, self.bias
        # print(x.shape)
        out = F.linear(input=x, weight=weight, bias=bias)
        return out


class WordEmbed(nn.Module):
    def __init__(self, input_shape, num_filters):
        super(WordEmbed, self).__init__()
        b, c = input_shape

        self.weights = nn.Parameter(torch.ones(num_filters, c))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            weight = params["weights"]
        else:
            pass
            #print('no inner loop params', self)

            weight = self.weights
        # print(x.shape)
        out = F.linear(input=x, weight=weight)
        return out

class MetaBatchNormLayer(nn.Module):
    def __init__(self, num_features, device, args, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, meta_batch_norm=True, no_learnable_params=False,
                 use_per_step_bn_statistics=False):
        """
        A MetaBatchNorm layer. Applies the same functionality of a standard BatchNorm layer with the added functionality of
        being able to receive a parameter dictionary at the forward pass which allows the convolution to use external
        weights instead of the internal ones stored in the conv layer. Useful for inner loop optimization in the meta
        learning setting. Also has the additional functionality of being able to store per step running stats and per step beta and gamma.
        :param num_features:
        :param device:
        :param args:
        :param eps:
        :param momentum:
        :param affine:
        :param track_running_stats:
        :param meta_batch_norm:
        :param no_learnable_params:
        :param use_per_step_bn_statistics:
        """
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.device = device
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args
        self.learnable_gamma = self.args.learnable_bn_gamma
        self.learnable_beta = self.args.learnable_bn_beta

        if use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                             requires_grad=False)
            self.running_var = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                            requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features),
                                       requires_grad=self.learnable_gamma)
        else:
            self.running_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.running_var = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        if self.args.enable_inner_loop_optimizable_bn_params:
            self.bias = nn.Parameter(torch.zeros(num_features),
                                     requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features),
                                       requires_grad=self.learnable_gamma)

        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)

        self.momentum = momentum

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
        Forward propagates by applying a bach norm function. If params are none then internal params are used.
        Otherwise passed params will be used to execute the function.
        :param input: input data batch, size either can be any.
        :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
         collecting per step batch statistics. It indexes the correct object to use for the current time-step
        :param params: A dictionary containing 'weight' and 'bias'.
        :param training: Whether this is currently the training or evaluation phase.
        :param backup_running_statistics: Whether to backup the running statistics. This is used
        at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
        :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight, bias) = params["weight"], params["bias"]
            #print(num_step, params['weight'])
        else:
            #print(num_step, "no params")
            weight, bias = self.weight, self.bias

        if self.use_per_step_bn_statistics:
            running_mean = self.running_mean[num_step]
            running_var = self.running_var[num_step]
            if params is None:
                if not self.args.enable_inner_loop_optimizable_bn_params:
                    bias = self.bias[num_step]
                    weight = self.weight[num_step]
        else:
            running_mean = None
            running_var = None


        if backup_running_statistics and self.use_per_step_bn_statistics:
            self.backup_running_mean.data = copy(self.running_mean.data)
            self.backup_running_var.data = copy(self.running_var.data)

        momentum = self.momentum

        output = F.batch_norm(input, running_mean, running_var, weight, bias,
                              training=True, momentum=momentum, eps=self.eps)

        return output

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        if self.use_per_step_bn_statistics:
            self.running_mean = nn.Parameter(self.backup_running_mean.to(device=self.device), requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var.to(device=self.device), requires_grad=False)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class MetaLayerNormLayer(nn.Module):
    def __init__(self, input_feature_shape, eps=1e-5, elementwise_affine=True):
        """
        A MetaLayerNorm layer. A layer that applies the same functionality as a layer norm layer with the added
        capability of being able to receive params at inference time to use instead of the internal ones. As well as
        being able to use its own internal weights.
        :param input_feature_shape: The input shape without the batch dimension, e.g. c, h, w
        :param eps: Epsilon to use for protection against overflows
        :param elementwise_affine: Whether to learn a multiplicative interaction parameter 'w' in addition to
        the biases.
        """
        super(MetaLayerNormLayer, self).__init__()
        if isinstance(input_feature_shape, numbers.Integral):
            input_feature_shape = (input_feature_shape,)
        self.normalized_shape = torch.Size(input_feature_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*input_feature_shape), requires_grad=False)
            self.bias = nn.Parameter(torch.Tensor(*input_feature_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset parameters to their initialization values.
        """
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying a layer norm function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            bias = params["bias"]
        else:
            bias = self.bias
            #print('no inner loop params', self)

        return F.layer_norm(
            input, self.normalized_shape, self.weight, bias, self.eps)

    def restore_backup_stats(self):
        pass

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
class MetaConvNormLayerReLU(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False, device=None):
        """
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param args: A named tuple containing the system's hyperparameters.
           :param device: The device to run the layer on.
           :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
           :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
           meta-conv etc.
           :param input_shape: The image input shape in the form (b, c, h, w)
           :param num_filters: number of filters for convolutional layer
           :param kernel_size: the kernel size of the convolutional layer
           :param stride: the stride of the convolutional layer
           :param padding: the bias of the convolutional layer
           :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaConvNormLayerReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros(self.input_shape)

        out = x

        self.conv = MetaConv2dLayer(in_channels=out.shape[1], out_channels=self.num_filters,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding)



        out = self.conv(out)

        if self.normalization:
            if self.args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(out.shape[1], track_running_stats=True,
                                                     meta_batch_norm=self.meta_layer,
                                                     no_learnable_params=self.no_bn_learnable_params,
                                                     device=self.device,
                                                     use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                     args=self.args)
            elif self.args.norm_layer == "layer_norm":
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])

            out = self.norm_layer(out, num_step=0)

        out = F.leaky_relu(out)

        print(out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        batch_norm_params = None
        conv_params = None
        activation_function_pre_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']

                if 'activation_function_pre' in params:
                    activation_function_pre_params = params['activation_function_pre']

            conv_params = params['conv']

        out = x


        out = self.conv(out, params=conv_params)

        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          backup_running_statistics=backup_running_statistics)

        out = F.leaky_relu(out)

        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()


class MetaNormLayerConvReLU(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, args, normalization=True,
                 meta_layer=True, no_bn_learnable_params=False, device=None):
        """
           Initializes a BatchNorm->Conv->ReLU layer which applies those operation in that order.
           :param args: A named tuple containing the system's hyperparameters.
           :param device: The device to run the layer on.
           :param normalization: The type of normalization to use 'batch_norm' or 'layer_norm'
           :param meta_layer: Whether this layer will require meta-layer capabilities such as meta-batch norm,
           meta-conv etc.
           :param input_shape: The image input shape in the form (b, c, h, w)
           :param num_filters: number of filters for convolutional layer
           :param kernel_size: the kernel size of the convolutional layer
           :param stride: the stride of the convolutional layer
           :param padding: the bias of the convolutional layer
           :param use_bias: whether the convolutional layer utilizes a bias
        """
        super(MetaNormLayerConvReLU, self).__init__()
        self.normalization = normalization
        self.use_per_step_bn_statistics = args.per_step_bn_statistics
        self.input_shape = input_shape
        self.args = args
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.meta_layer = meta_layer
        self.no_bn_learnable_params = no_bn_learnable_params
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_block()

    def build_block(self):

        x = torch.zeros(self.input_shape)

        out = x
        if self.normalization:
            if self.args.norm_layer == "batch_norm":
                self.norm_layer = MetaBatchNormLayer(self.input_shape[1], track_running_stats=True,
                                                     meta_batch_norm=self.meta_layer,
                                                     no_learnable_params=self.no_bn_learnable_params,
                                                     device=self.device,
                                                     use_per_step_bn_statistics=self.use_per_step_bn_statistics,
                                                     args=self.args)
            elif self.args.norm_layer == "layer_norm":
                self.norm_layer = MetaLayerNormLayer(input_feature_shape=out.shape[1:])

            out = self.norm_layer.forward(out, num_step=0)
        self.conv = MetaConv2dLayer(in_channels=out.shape[1], out_channels=self.num_filters,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride, padding=self.padding)


        self.layer_dict['activation_function_pre'] = nn.LeakyReLU()


        out = self.layer_dict['activation_function_pre'].forward(self.conv.forward(out))
        print(out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False):
        """
            Forward propagates by applying the function. If params are none then internal params are used.
            Otherwise passed params will be used to execute the function.
            :param input: input data batch, size either can be any.
            :param num_step: The current inner loop step being taken. This is used when we are learning per step params and
             collecting per step batch statistics. It indexes the correct object to use for the current time-step
            :param params: A dictionary containing 'weight' and 'bias'.
            :param training: Whether this is currently the training or evaluation phase.
            :param backup_running_statistics: Whether to backup the running statistics. This is used
            at evaluation time, when after the pass is complete we want to throw away the collected validation stats.
            :return: The result of the batch norm operation.
        """
        batch_norm_params = None

        if params is not None:
            params = extract_top_level_dict(current_dict=params)

            if self.normalization:
                if 'norm_layer' in params:
                    batch_norm_params = params['norm_layer']

            conv_params = params['conv']
        else:
            conv_params = None
            #print('no inner loop params', self)

        out = x

        if self.normalization:
            out = self.norm_layer.forward(out, num_step=num_step,
                                          params=batch_norm_params, training=training,
                                          backup_running_statistics=backup_running_statistics)

        out = self.conv.forward(out, params=conv_params)
        out = self.layer_dict['activation_function_pre'].forward(out)

        return out

    def restore_backup_stats(self):
        """
        Restore stored statistics from the backup, replacing the current ones.
        """
        if self.normalization:
            self.norm_layer.restore_backup_stats()

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        # Assumes 1 layer RNN and that the input size is same as hidden size
        super(SimpleRNN, self).__init__()
        assert(input_size == hidden_size)
        b, c = input_size
        num_filters = 256
        self.weights1 = nn.Parameter(torch.ones(num_filters,c))
        nn.init.xavier_uniform_(self.weights1)
        self.bias1 = nn.Parameter(torch.zeros(num_filters))
        self.weights2 = nn.Parameter(torch.ones(num_filters,c))
        nn.init.xavier_uniform_(self.weights2)
        self.bias2 = nn.Parameter(torch.zeros(num_filters))

        self.hidden_size = hidden_size[1]
        # self.feedback = nn.Linear(hidden_size, hidden_size)
        # self.nonlinearity = nn.ReLU(inplace=True)


    def forward(self, x_t, h_t_1=None,params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            (weight1,bias1) = params["weights1"], params["bias1"]
            (weight2,bias2) = params["weights2"], params["bias2"]
        else:
            pass
            weight1, bias1 = self.weights1, self.bias1
            weight2, bias2 = self.weights2, self.bias2
        if h_t_1 is None:
            batch_size = x_t.size(0)
            h_t_1 = torch.autograd.Variable(x_t.data.new(batch_size, self.hidden_size).zero_())

        # output = x_t + self.feedback(h_t_1)
        output = F.linear(input=x_t,weight=weight1,bias=bias1) + F.linear(input=h_t_1,weight=weight2,bias=bias2)
        output = F.relu(output)
        # output = F.leaky_relu(output,0.01)
        return output

class VGGReLUNormNetwork(nn.Module):
    def __init__(self, im_shape, num_output_classes, args, device, meta_classifier=True):
        """
        Builds a multilayer convolutional network. It also provides functionality for passing external parameters to be
        used at inference time. Enables inner loop optimization readily.
        :param im_shape: The input image batch shape.
        :param num_output_classes: The number of output classes of the network.
        :param args: A named tuple containing the system's hyperparameters.
        :param device: The device to run this on.
        :param meta_classifier: A flag indicating whether the system's meta-learning (inner-loop) functionalities should
        be enabled.
        """
        super(VGGReLUNormNetwork, self).__init__()
        b, c, self.h, self.w = im_shape
        self.device = device
        self.total_layers = 0
        self.args = args
        self.upscale_shapes = []
        self.cnn_filters = args.cnn_num_filters
        self.input_shape = list(im_shape)
        self.num_stages = args.num_stages
        self.num_output_classes = num_output_classes
        self.kernel_size = args.kernel_size
        
        self.writer = SummaryWriter('logs')
        
        if args.max_pooling:
            print("Using max pooling")
            self.conv_stride = 1
        else:
            print("Using strided convolutions")
            self.conv_stride = 2
        self.meta_classifier = meta_classifier

        self.build_network()
        print("meta network params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):
        """
        Builds the network before inference is required by creating some dummy inputs with the same input as the
        self.im_shape tuple. Then passes that through the network and dynamically computes input shapes and
        sets output shapes for each layer.
        """
        x = torch.zeros(self.input_shape)
        input_loc_size = self.args.A
        # input_loc_size = 2
        loc = torch.zeros(self.input_shape[0],input_loc_size)
        hidden = None
        input_act_size = 256
        if self.args.actOnElev:
            input_act_size += 1
        if self.args.actOnAzim:
            input_act_size += 1
        if self.args.actOnTime:
            input_act_size += 1
        out = x

        self.layer_dict = nn.ModuleDict()
        self.upscale_shapes.append(x.shape)

        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)] = MetaConvNormLayerReLU(input_shape=out.shape,
                                                                        num_filters=self.cnn_filters,
                                                                        kernel_size=self.kernel_size, stride=self.conv_stride,
                                                                        padding=self.args.conv_padding,
                                                                        args=self.args, normalization=True,
                                                                        meta_layer=self.meta_classifier,
                                                                        no_bn_learnable_params=False,
                                                                        device=self.device)
            out = self.layer_dict['conv{}'.format(i)](out, training=True, num_step=0)

            if self.args.max_pooling:
                out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)


        if not self.args.max_pooling:
            out = F.avg_pool2d(out, out.shape[2])

        # (1) feature encoder
        self.encoder_features_shape = list(out.shape)
        out = out.view(out.shape[0], -1)

        # (2) sense motion
        self.layer_dict['sense_motion'] = WordEmbed(input_shape=(out.shape[0],input_loc_size),
                                                          num_filters=128) # input_size_loc = 2
        y = self.layer_dict['sense_motion'](loc)
        # y = F.relu(self.layer_dict['sense_motion'](loc))

        # (3) fuse the feature vector and sense location vector to give a fusion vector
        self.layer_dict['fuse'] = MetaLinearLayer(input_shape=(out.shape[0],out.shape[1]+self.args.embed_size),
                                                  num_filters=256)
        out = torch.cat([out,y],dim=1)
        out = self.layer_dict['fuse'](out)
        # out = MetaLayerNormLayer(input_feature_shape=z.shape[1:])

        # (4) aggregate
        self.layer_dict['aggregate'] = SimpleRNN(input_size=(out.shape[0],256),hidden_size=(out.shape[0],256))
        
        out = self.layer_dict['aggregate'](out,hidden)

        # (5) act module
        if self.args.actorType == 'actor':
            act = out 
            self.layer_dict['act'] = MetaLinearLayer(input_shape=(out.shape[0],input_act_size),num_filters=self.args.A)
            if self.args.actOnElev:
                act = torch.cat([act,torch.zeros(out.shape[0],1)],dim=1)
            if self.args.actOnAzim:
                act = torch.cat([act,torch.zeros(out.shape[0],1)],dim=1)
            if self.args.actOnTime:
                act = torch.cat([act,torch.zeros(out.shape[0],1)],dim=1)
            act = self.layer_dict['act'](act)
        

        
        # (6) classifier
        self.layer_dict['classifier'] = MetaLinearLayer(input_shape=out.shape,num_filters=self.num_output_classes)
        out = self.layer_dict['classifier'](out)

        print("VGGNetwork build", out.shape)

    def forward(self, x, num_step, params=None, training=False, backup_running_statistics=False,eval_opts=None,current_iter=None,sample_idx=None,task_id=None):
        """
        Forward propages through the network. If any params are passed then they are used instead of stored params.
        :param x: Input image batch.
        :param num_step: The current inner loop step number
        :param params: If params are None then internal parameters are used. If params are a dictionary with keys the
         same as the layer names then they will be used instead.
        :param training: Whether this is training (True) or eval time.
        :param backup_running_statistics: Whether to backup the running statistics in their backup store. Which is
        then used to reset the stats back to a previous state (usually after an eval loop, when we want to throw away stored statistics)
        :return: Logits of shape b, num_output_classes.
        """
        param_dict = dict()

        batch_size = x.views.shape[0]
        visited_idxes = []
        reward_all = []
        log_prob_act_all = []
        entropy_all = []
        baseline_all = []
        action_probs_all = []
        hidden = Variable(torch.zeros(batch_size,256))
        R_avg_expert = 0
        avg_count_expert = 0

        if self.args.use_cuda:
            hidden = hidden.cuda()
        if self.args.T > 1:
            actions_taken = torch.zeros(batch_size,self.args.T-1)
        else:
            actions_taken = None

        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)

        # print('top network', param_dict.keys())
        for name, param in self.layer_dict.named_parameters():
            path_bits = name.split(".")
            layer_name = path_bits[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None

        # out = x
        
        for t in range(self.args.T):
            im, delta, pro = x.get_view()
            ##############################one-hot delta########################################
            delta = [tuple(i) for i in delta]
            delta = [self.args.delta_to_act[i] for i in delta]
            delta = torch.zeros(im.shape[0],self.args.A).scatter_(1, torch.LongTensor(np.expand_dims(delta,1)),1)
            ##############################one-hot delta########################################
            im, delta, pro = torch.Tensor(im), torch.Tensor(delta), torch.Tensor(pro)
            if self.args.use_cuda:
                im, delta, pro = im.cuda(), delta.cuda(), pro.cuda()
                im, delta, pro = Variable(im), Variable(delta), Variable(pro)
            
            visited_idxes.append(x.idx)
            
            if self.args.actOnTime:
                time = torch.Tensor([[t] for i in range(batch_size)])
                if self.args.use_cuda:
                    time = time.cuda()
                time = Variable(time)
            out = im
            for i in range(self.num_stages):
                out = self.layer_dict['conv{}'.format(i)](out, params=param_dict['conv{}'.format(i)], training=training,
                                                        backup_running_statistics=backup_running_statistics,
                                                        num_step=num_step)
                if self.args.max_pooling:
                    out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)

            if not self.args.max_pooling:
                out = F.avg_pool2d(out, out.shape[2])

            out = out.view(out.size(0), -1)
            loc = self.layer_dict['sense_motion'](delta, param_dict['sense_motion'])
            # loc = F.relu(loc)
            out = torch.cat([out,loc],dim=1)
            out = self.layer_dict['fuse'](out, param_dict['fuse'])
            out = F.relu(out)

            hidden = self.layer_dict['aggregate'](out, hidden, param_dict['aggregate'])
            if self.args.actorType == 'actor':
                act_input = hidden.view(batch_size,-1)
    
                if self.args.actOnElev:
                    act_input = torch.cat([act_input,pro[:,0].contiguous().view(-1,1)],dim=1)
                if self.args.actOnAzim:
                    act_input = torch.cat([act_input,pro[:,1].contiguous().view(-1,1)],dim=1)
                if self.args.actOnTime:
                    act_input = torch.cat([act_input,time],dim=1)

                act = self.layer_dict['act'](act_input, param_dict['act'])
                # act_prob = F.hardtanh(input=act,min_val=0,max_val=1,inplace=True)
                # action_probs = F.normalize(act_prob+1e-8,p=1,dim=1)
                action_probs = F.softmax(act,dim=1)
                # print('action_probs',action_probs)
            else:
                action_probs = None
                act_input = None


            # hidden_normalized = F.normalize(hidden.view(batch_size,-1),p=1,dim=1)
            hidden_normalized = hidden.view(batch_size,-1)
            if t > 0:
                enable_rewards = False
                if t == self.args.T - 1:
                    enable_rewards = True
                if enable_rewards:
                    classifier_activations = self.layer_dict['classifier'](hidden_normalized, param_dict['classifier'])
                    # classifier_activations = F.softmax(classifier_activations,dim=1)
                    reward = x.reward_fn(classifier_activations.data.cpu().numpy())
                    reward = torch.Tensor(reward)
                    # print(reward)
                    if self.args.use_cuda:
                        reward = reward.cuda()
                    reward_all[t-1] += reward

                    # baseline_class_pred = np.zeros((batch_size,self.args.num_classes_per_set))
                    # baseline = x.reward_fn(baseline_class_pred)
                    # baseline = torch.Tensor(baseline)
                    # if self.args.use_cuda:
                        # baseline = baseline.cuda()
                # else:
                    # baseline = torch.zeros(batch_size)
                    # if self.args.use_cuda:
                        # baseline = baseline.cuda()
                # baseline_all.append(baseline)

            if t < self.args.T - 1:
                if self.args.actorType == 'actor':
                    # if eval_opts == None:
                    #     act = action_probs.multinomial(num_samples=1).data
                    # else:
                    _,act = action_probs.max(dim=1)
                    act = act.data.view(-1,1)
                
                    entropy = -(action_probs*((action_probs+1e-7).log())).sum(dim=1)
                    log_prob_act = (action_probs[range(act.size(0)),act[:,0]]+1e-7).log()
                elif self.args.actorType == 'random':
                    act = torch.Tensor(np.random.randint(0,self.args.A,size=(batch_size,1)))
                    log_prob_act = None
                    entropy = None
                
                actions_taken[:,t] = act[:,0]
                # if t == self.args.T-2:
                #     import matplotlib.pyplot as plt
                #     plt.hist(actions_taken.view(-1))
                #     plt.show()

                reward_expert = x.rotate(act[:,0])
                reward_expert = torch.Tensor(reward_expert)
                if self.args.use_cuda:
                    reward_expert = reward_expert.cuda()
                
                # R_avg_expert = (R_avg_expert * avg_count_expert + reward_expert.sum()) / (avg_count_expert + batch_size)
                # avg_count_expert += batch_size

                reward_all.append(reward_expert)
                log_prob_act_all.append(log_prob_act)
                entropy_all.append(entropy)
                action_probs_all.append(action_probs)
        
        visualize(x, visited_idxes,action_probs_all,self.args,sample_idx=sample_idx,task_id=task_id)
        # Classification loss
        loss = Variable(torch.Tensor([0]))
        R = torch.zeros(batch_size)
        # B = torch.zeros(batch_size)
        if self.args.use_cuda:
            R = R.cuda()
            # B = B.cuda()
            loss = loss.cuda()
        labs = x.labs
        if self.args.use_cuda:
            labs = labs.cuda()
        labs = Variable(labs)
        criterion = nn.CrossEntropyLoss()
        loss_c = criterion(classifier_activations, labs)
        # loss = loss + criterion(classifier_activations, labs)
        # loss = loss + F.cross_entropy(input=classifier_activations, target=labs)
        clsf_loss = loss_c.data.cpu()
        # print('loss',loss.data)
        # REINFORCE loss based on T-1 transitions

        loss_pg = 0
        loss_ent = 0
        for t in reversed(range(self.args.T-1)):
            if self.args.actorType == 'actor':
                R = R + reward_all[t]
                # B = B + baseline_all[t] + R_avg_expert
                # adv = R - B
                loss_term_1 = - self.args.lambda_reward * (log_prob_act_all[t]*Variable(R,requires_grad=False)*self.args.reward_scale).sum()/batch_size
                loss_pg += loss_term_1
                loss_term_2 = - self.args.lambda_entropy * entropy_all[t].sum()/batch_size
                loss_ent += loss_term_2
        
        
        loss_pg = loss_pg/(self.args.T-1)
        loss_ent = loss_ent/(self.args.T-1)
        loss = loss_c + loss_pg + loss_ent


        # if training and x.labs.shape == torch.Size([75]):
        #     # print('total_loss',loss.data)
        #     print('clsf_loss',clsf_loss.data)
        #     print('pg_loss',pg_loss.data)
            # print('ent_loss',ent_loss.data)
        #     self.writer.add_scalar('total_loss',loss,current_iter)
        #     self.writer.add_scalar('clsf_loss',clsf_loss,current_iter)
        #     self.writer.add_scalar('pg_loss',pg_loss,current_iter)
        #     self.writer.add_scalar('ent_loss',ent_loss,current_iter)    
        # out = self.layer_dict['linear'](out, param_dict['linear'])

        # return out
        # return probs, hidden
        return classifier_activations, loss, loss_c, loss_pg, loss_ent, visited_idxes

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for i in range(self.num_stages):
            self.layer_dict['conv{}'.format(i)].restore_backup_stats()


