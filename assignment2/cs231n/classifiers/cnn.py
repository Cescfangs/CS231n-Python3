import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
        self.params['W2'] = np.random.randn(num_filters * H * W // 4, hidden_dim) * weight_scale
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_out, conv_cahce = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        hidden_out, hidden_cache = affine_relu_forward(conv_out, W2, b2)
        scores, scores_cache = affine_forward(hidden_out, W3, b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        loss += self.reg * 0.5 * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))

        dscore, dW3, db3 = affine_backward(dout, scores_cache)
        drelu, dW2, db2 = affine_relu_backward(dscore, hidden_cache)
        dx, dW1, db1 = conv_relu_pool_backward(drelu, conv_cahce)
        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['W3'] = dW3
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3
        # grads['dx'] = dx

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class CNN(object):
    """
    A self made convolutional network with arbitrary architecture:

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, architecture=None, input_dim=(3, 32, 32), filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        if architecture is None:
            self.architecture = [{'type': 'conv_relu_pool', 'repeat': 1, 'num_filters': 32,
                                  'filter_size': 7}, {'type': 'affine', 'repeat': [100, 10]}]
        else:
            self.architecture = architecture

        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim

        prev_dim = C
        conv_out_width = H
        conv_out_height = W
        count = 0
        for achi in self.architecture:
            if achi['type'][:4] == 'conv':
                num_filters = achi['num_filters']
                if(achi['type'][-4:] == 'pool'):
                    conv_out_height = conv_out_height // (2**achi['repeat'])
                    conv_out_width = conv_out_width // (2**achi['repeat'])
                for i in range(achi['repeat']):
                    self.params['W' + str(count)] = np.random.randn(achi['num_filters'], prev_dim,
                                                                    filter_size, filter_size) * weight_scale
                    self.params['b' + str(count)] = np.zeros(achi['num_filters'])
                    prev_dim = achi['num_filters']
                    count += 1

            else:
                prev_dim = conv_out_height * conv_out_width * num_filters
                for dim in achi['repeat']:
                    self.params['W' + str(count)] = np.random.randn(prev_dim, dim) * weight_scale
                    self.params['b' + str(count)] = np.zeros(dim)
                    prev_dim = dim
                    count += 1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = self.params['W0'].shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the arbitrary convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        count = 0
        caches = []
        out = X
        for achi in self.architecture:
            if achi['type'][:4] == 'conv':
                for i in range(achi['repeat']):
                    if(achi['type'][-4:] == 'pool'):
                        out, cache = conv_relu_pool_forward(
                            out, self.params['W' + str(count)], self.params['b' + str(count)], conv_param, pool_param)
                        caches.append(cache)
                    else:
                        out, cache = conv_relu_forward(
                            out, self.params['W' + str(count)], self.params['b' + str(count)], conv_param)
                        caches.append(cache)
                    count += 1

            else:
                for dim in achi['repeat'][:-1]:
                    out, cache = affine_relu_forward(out, self.params['W' + str(count)], self.params['b' + str(count)])
                    count += 1
                    caches.append(cache)

                scores, scores_cache = affine_forward(
                    out,  self.params['W' + str(count)], self.params['b' + str(count)])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################

        loss, dout = softmax_loss(scores, y)
        for i in range(count + 1):
            loss += self.reg * 0.5 * (np.sum(np.square(self.params['W' + str(i)])))

        dscore, grads['W' + str(count)], grads['b' + str(count)] = affine_backward(dout, scores_cache)
        count -= 1
        dx = dscore
        for achi in self.architecture[-1::-1]:
            if achi['type'][:4] == 'conv':
                for i in range(achi['repeat']):
                    if(achi['type'][-4:] == 'pool'):
                        dx, grads['W' + str(count)], grads['b' + str(count)
                                                           ] = conv_relu_pool_backward(dx, caches[count])
                    else:
                        dx, grads['W' + str(count)], grads['b' + str(count)] = conv_relu_backward(dx, caches[count])
                    count -= 1

            else:
                for dim in achi['repeat'][-2::-1]:
                    dx, grads['W' + str(count)], grads['b' + str(count)] = affine_relu_backward(dx, caches[count])
                    count -= 1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
