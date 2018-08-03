
import cv2
import numpy as np
import math
from export_env_variables import *
from defs import *
# from utils import *

def visualize_kernels(net, layer, zoom=5, filename=""):
    """
    Visualize kernels in the given convolutional layer.

    :param net: caffe network
    :type net: caffe.Net
    :param layer: layer name
    :type layer: string
    :param zoom: the number of pixels (in width and height) per kernel weight
    :type zoom: int
    :return: image visualizing the kernels in a grid
    :rtype: numpy.ndarray
    """

    num_kernels = net.params[layer][0].data.shape[0]
    num_channels = net.params[layer][0].data.shape[1]
    kernel_height = net.params[layer][0].data.shape[2]
    kernel_width = net.params[layer][0].data.shape[3]

    image = np.zeros((num_kernels * zoom * kernel_height, num_channels * zoom * kernel_width))
    for k in range(num_kernels):
        for c in range(num_channels):
            kernel = net.params[layer][0].data[k, c, :, :]
            kernel = cv2.resize(kernel, (zoom * kernel_height, zoom * kernel_width), kernel, 0, 0, cv2.INTER_NEAREST)
            kernel = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))
            image[k * zoom * kernel_height:(k + 1) * zoom * kernel_height,
            c * zoom * kernel_width:(c + 1) * zoom * kernel_width] = kernel




            # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image, cmap='gray', interpolation='nearest')


    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    return image
# -------------------------------------------------------------------------------------------------------


def vis_square(data, padsize=1, padval=0, filename=""):
    data -= data.min()
    if data.max() != 0:
        data /= data.max()
                
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
                                    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
                        
    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(data, cmap='gray', interpolation='nearest')

    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

# -------------------------------------------------------------------------------------------------------

def show_blobs(data, filename=""):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    if data.max() != data.min():
        data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])


    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(data, cmap='gray', interpolation='nearest')

    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    else:
        plt.show()

# -------------------------------------------------------------------------------------------------------


def visualize_conv_layers(net, layer_name, padding=4, filename=''):
    # The parameters are a list of [weights, biases]
    data = np.copy(net.params[layer_name][0].data)
    # N is the total number of convolutions
    N = data.shape[0] * data.shape[1]
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    filter_size = data.shape[2]
    # Size of the result image including padding
    result_size = filters_per_row * (filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size, result_size))

    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    for n in range(data.shape[0]):
        for c in range(data.shape[1]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for i in range(filter_size):
                for j in range(filter_size):
                    result[filter_y * (filter_size + padding) + i, filter_x * (filter_size + padding) + j] = data[
                        n, c, i, j]
            filter_x += 1

    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)

    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result, cmap='gray', interpolation='nearest')

    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    else:
        plt.show()
# -------------------------------------------------------------------------------------------------------

def visualize_fc_layers(net, layer_name, padding=4, filename=''):
    # The parameters are a list of [weights, biases]
    data = np.copy(net.params[layer_name][0].data)
    # # N is the total number of convolutions
    N = 1
    # # Ensure the resulting image is square
    filters_per_row = 1
    # Assume the filters are square
    filter_size = data.shape[0]
    # Size of the result image including padding
    result_size = filters_per_row * (filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size, result_size))

    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0

    for i in range(filter_size):
        for j in range(filter_size):
            result[filter_y * (filter_size + padding) + i, filter_x * (filter_size + padding) + j] = data[
                i, j]


    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)

    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result, interpolation='nearest')

    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    # plt.show()

# -------------------------------------------------------------------------------------------------------
