import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    score = X.dot(W)
    score = score - score.max()
    p = np.zeros(score.shape)
    li = np.zeros_like(score)
    for i in range(score.shape[0]):
        p_sum = np.sum(np.exp(score[i, :]))
        for j in range(score.shape[1]):
            p[i, j] = np.exp(score[i, j]) / p_sum
            li[i, j] = - np.sum(np.log(p[i, j]) * (y[i] == j))
    loss = li.sum() / li.shape[0]
    loss += 0.5 * reg * np.sum(W * W)
    # print('ypred:', score.shape, 'p', p.shape, 'w', W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            dW[i, j] = np.sum((p[:, j] - (y[:] == j)) * X[:, i]) / len(X)
    dW += W * reg
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    score = X.dot(W)
    score -= np.max(score, axis=1, keepdims=True)
    # p = np.zeros_like(score)
    p = np.exp(score) / np.sum(np.exp(score), axis=1, keepdims=True)
    loss = - np.sum(np.log(p[np.arange(len(X)), y])) / len(X) + 0.5 * reg * np.sum(np.square(W))
    y_id = np.zeros_like(p)
    y_id[np.arange(len(X)), y] = 1
    dW = X.T.dot(p - y_id) / len(X) + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
