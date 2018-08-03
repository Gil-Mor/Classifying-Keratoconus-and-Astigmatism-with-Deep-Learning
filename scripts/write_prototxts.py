import sys
import os
from export_env_variables import *
from defs import *



DOUBLE_QOUTES = "\""

FULLY_CONNECTED_LAYER_NAME = "fc8_kc"



def write_prototxts(mode, solver_net_params):
    """
    :type mode: Mode
    :type solver_net_params: Solver_Net_parameters
    """

    write_net_prototxt(mode, solver_net_params)

    if solver_net_params.solver_type==ADAM_SOLVER_TYPE:
        write_Adam_Solver(mode, solver_net_params)
    else:
        write_solver_prototxt(mode, solver_net_params)


    write_deploy_prototxt(mode)
# -------------------------------------------------------------------------------------

def write_Adam_Solver(mode, solver_net_params):
    """
    :type mode: Mode
    :type solver_net_params: Solver_Net_parameters
    """

    try:
        os.makedirs(os.path.dirname(mode.solver_prototxt))
    except:
        pass

    file = open(mode.solver_prototxt, "w")

    s = """net: """ + DOUBLE_QOUTES + mode.train_val_prototxt + DOUBLE_QOUTES + """
test_initialization: true # Good for plotting. Uncomment if you want to get rid of the testing in iteration 0
test_iter: """ + str(solver_net_params.test_iter) + """
test_interval: """ + str(solver_net_params.test_interval) + """
base_lr: 0.0001 # was 0.001
momentum: 0.9
momentum2: 0.999
# since Adam dynamically changes the learning rate, we set the base learning
# rate to a fixed value
lr_policy: "fixed"
display: """ + str(solver_net_params.display_iter) + """ # Display training info every X iterations.
max_iter: """ + str(solver_net_params.max_iter) + """
snapshot: """ + str(solver_net_params.snapshot_iter) + """
snapshot_prefix: """ + DOUBLE_QOUTES + mode.snapshot_prefix_forsolver_prototxt + DOUBLE_QOUTES + """
type: "Adam"
"""

    if PLATFORM == EC2_GPU_Platform:
        s += "solver_mode: GPU\n"
    else:
        s += "solver_mode: CPU\n"

    file.write(s)

    file.close()
# -------------------------------------------------------------------------


def write_solver_prototxt(mode, solver_net_params):
    """
    :type mode: Mode
    :type solver_net_params: Solver_Net_parameters
    """

    try:
        os.makedirs(os.path.dirname(mode.solver_prototxt))
    except:
        pass

    file = open(mode.solver_prototxt, "w")

    s = """net: """ + DOUBLE_QOUTES + mode.train_val_prototxt + DOUBLE_QOUTES
    if solver_net_params.test_iter:
        s += """
test_initialization: true # Good for plotting. Uncomment if you want to get rid of the testing in iteration 0
test_iter: """ + str(solver_net_params.test_iter) + """
test_interval: """ + str(solver_net_params.test_interval)
    else:
        s += """
test_initialization: false     
base_lr: """ + str(solver_net_params.lr) + """
lr_policy: "step" # Drop lr by a factor of gamma in a linear manner - lr *= gamma every 'stepsize'
gamma: 0.1 # Drop learning rate by x factor.
stepsize: """ + str(solver_net_params.lr_stepsize) + """ # Drop learning rate after x iterations
display: """ + str(solver_net_params.display_iter) + """ # Display training info every X iterations.
max_iter: """ + str(solver_net_params.max_iter) + """
momentum: """ + str(solver_net_params.momentum) + """ # kill of momentum in steep curves. was 0.9
weight_decay: """ + str(solver_net_params.weight_decay) + """ # Regurelize layers weights. Higher for fewer samples. usually between 1e-6 and 1e-4. Was 0.0005
snapshot: """ + str(solver_net_params.snapshot_iter) + """
snapshot_prefix: """ + DOUBLE_QOUTES + mode.snapshot_prefix_forsolver_prototxt + DOUBLE_QOUTES + """
"""

    if PLATFORM == EC2_GPU_Platform:
        s += "solver_mode: GPU\n"
    else:
        s += "solver_mode: CPU\n"


    file.write(s)

    file.close()
# -------------------------------------------------------------------------



def write_net_prototxt(mode, solver_net_params):
    """
    :type mode: Mode
    :type solver_net_params: Solver_Net_parameters
    """

    try:
        os.makedirs(os.path.dirname(mode.train_val_prototxt))
    except:
        pass

    file = open(mode.train_val_prototxt, "w")

    file.write(

"""name: "CaffeNet"
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 227
    mean_file: """ + DOUBLE_QOUTES + mode.mean_binaryproto + DOUBLE_QOUTES + """
  }
  data_param {
    source: """ + DOUBLE_QOUTES + mode.train_lmdb + DOUBLE_QOUTES + """
    batch_size: """ + str(solver_net_params.train_batch_size) + """
    backend: LMDB

  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 227
    mean_file: """ + DOUBLE_QOUTES + mode.mean_binaryproto  + DOUBLE_QOUTES + """
  }
  data_param {
    source: """ + DOUBLE_QOUTES + mode.val_lmdb + DOUBLE_QOUTES + """
    batch_size: """ + str(solver_net_params.val_batch_size) + """
    backend: LMDB
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "pool1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "norm1"
  top: "conv2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm2"
  type: "LRN"
  bottom: "pool2"
  top: "norm2"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "norm2"
  top: "conv3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "conv4"
  type: "Convolution"
  bottom: "conv3"
  top: "conv4"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu4"
  type: "ReLU"
  bottom: "conv4"
  top: "conv4"
}
layer {
  name: "conv5"
  type: "Convolution"
  bottom: "conv4"
  top: "conv5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    group: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu5"
  type: "ReLU"
  bottom: "conv5"
  top: "conv5"
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "conv5"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
layer {
  name: "drop6"
  type: "Dropout"
  bottom: "fc6"
  top: "fc6"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc7"
  type: "InnerProduct"
  bottom: "fc6"
  top: "fc7"
  param {
    lr_mult: 5
    decay_mult: 1
  }
  param {
    lr_mult: 10
    decay_mult: 0
  }
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: "gaussian"
      std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 1
    }
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
layer {
  name: "drop7"
  type: "Dropout"
  bottom: "fc7"
  top: "fc7"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc8_kc"
  type: "InnerProduct"
  bottom: "fc7"
  top: "fc8_kc"
  param {
    lr_mult: 10
    decay_mult: 1
  }
  param {
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: """ + str(mode.get_num_of_classes()) + """
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "fc8_kc"
  bottom: "label"
  top: "accuracy"
  # Can also be in training. might force testing in each iteration?
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "fc8_kc"
  bottom: "label"
  top: "loss"
}
# Get predicted probabilities via test_nets[0].blobs['prob'].data
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc8_kc"
  top: "prob"
  include {
    phase: TEST
  }
}
# silence prob layer - bug - makes the layer unavailable as output from test_nets[0].forward. use test_nets[0].blobs['prob'].data
layer {
  name: "silence_prob"
  type: "Silence"
  bottom: "prob"
  include {
     phase: TEST
  }
}
#layer {
#  name: "prob_for_argmax"
#  type: "Softmax"
#  bottom: "fc8_kc"
#  top: "prob_for_argmax"
#  include {
#    phase: TEST
#  }
#}
#layer {
#  name: "argmax"
#  type: "ArgMax"
#  top: "argmax"
#  bottom: "prob_for_argmax"
#  argmax_param {
#    out_max_val: true
#  }
#  include {
#    phase: TEST
#  }
#}
# silence argmax layer
#layer {
#  name: "silence_argmax"
#  type: "Silence"
#  bottom: "argmax"
#  include {
#     phase: TEST
#  }
#}
""")

    file.close()
# -------------------------------------------------------------------------

def write_deploy_prototxt(mode):
    """

    :param mode:
    :type mode: Mode
    :return:
    """
    try:
        os.makedirs(os.path.dirname(mode.deploy_prototxt))
    except:
        pass

    file = open(mode.deploy_prototxt, "w")

    file.write(
"""name: "CaffeNet"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 227
      dim: 227
    }
  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "pool1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "norm1"
  top: "conv2"
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "norm2"
  type: "LRN"
  bottom: "pool2"
  top: "norm2"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "norm2"
  top: "conv3"
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "conv4"
  type: "Convolution"
  bottom: "conv3"
  top: "conv4"
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 2
  }
}
layer {
  name: "relu4"
  type: "ReLU"
  bottom: "conv4"
  top: "conv4"
}
layer {
  name: "conv5"
  type: "Convolution"
  bottom: "conv4"
  top: "conv5"
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    group: 2
  }
}
layer {
  name: "relu5"
  type: "ReLU"
  bottom: "conv5"
  top: "conv5"
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "conv5"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
# no need for drop out in testing. caffe skips this anyway. why it's here?
# layer {
#   name: "drop6"
#   type: "Dropout"
#   bottom: "fc6"
#   top: "fc6"
#   dropout_param {
#     dropout_ratio: 0.5
#   }
# }
layer {
  name: "fc7"
  type: "InnerProduct"
  bottom: "fc6"
  top: "fc7"
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
# no need for drop out in testing. caffe skips this anyway. why it's here?
# layer {
#   name: "drop7"
#   type: "Dropout"
#   bottom: "fc7"
#   top: "fc7"
#   dropout_param {
#     dropout_ratio: 0.5
#   }
# }
layer {
  name: "fc8_kc"
  type: "InnerProduct"
  bottom: "fc7"
  top: "fc8_kc"
  inner_product_param {
    num_output: """ + str(mode.get_num_of_classes()) + """
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc8_kc"
  top: "prob"
}
""")
    file.close()
# -------------------------------------------------------------------------


if __name__=="__main__":
    f = open(sys.rags[0], "w")
    net = sys.argv[1]
    iter = sys.argv[2]
    write_solver(net, iter, f)
    f.close()
