# ********************************************************** #
# Project 1 : CS-433 Machine Learning Class                  #
# Various regression models to detect Higgs particles        #
# Authors: Arthur Passuello, Francois Quellec, Julien Muster #
# ********************************************************** #

import numpy as np
from collections import deque

#####################################################
#             Loss Functions                        #
#####################################################
def compute_mse_loss(y, tx, w):
    """Calculate the loss using mse"""
    error = y - tx@w
    loss = error@error/(2*len(y))
    return loss

def compute_mae_loss(y, tx, w):
    """Calculate the loss using mae"""
    error = y - tx@w
    loss = np.mean(np.abs(error))
    return loss

def compute_log_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1 + np.exp(tx@w)) - y*(tx@w))
   
#####################################################
#              Activation Functions                 #
#####################################################
def sigmoid(t):
    """apply sigmoid activation function on t."""
    return 1./(1 + np.exp(-t))

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


#####################################################
#              Gradient Functions                   #
#####################################################
def calculate_least_square_gradient(y, tx, w):
    """Compute the gradient of the error for least square equation"""
    error = y - tx@w
    gradient =  -tx.T@error/len(y)
    return gradient

def compute_logistic_gradient(y, tx, w):
    """compute the gradient of loss for logistic regression"""
    return tx.T @ (sigmoid(tx@w) - y)

#####################################################
#              Models Definition                    #
#####################################################

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    loss = -1

    for n_iter in range(max_iters):
        # Compute loss and gradient then update accordingly the weights
        loss = compute_mse_loss(y, tx, w)
        gradient = calculate_least_square_gradient(y, tx, w)
        w -= gamma * gradient

    # Final loss
    loss = compute_mse_loss(y, tx, w)

    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    batch_size = 1
    w = initial_w
    loss = -1

    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        for _ in range(max_iters):
            # Compute loss and gradient then update accordingly the weights
            loss = compute_mse_loss(minibatch_y, minibatch_tx, w)
            gradient = calculate_least_square_gradient(minibatch_y, minibatch_tx, w)
            w -= gamma * gradient

    # Final loss
    loss = compute_mse_loss(y, tx, w)

    return w, loss

def least_squares(y, tx):
    """Least squares regression using normal equations"""
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    loss = compute_mse_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    omega = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    w = np.linalg.solve(tx.T@tx + omega, tx.T@y)
    loss = compute_mse_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent"""

     # init parameters
    threshold = 1e-8
    losses = collections.deque(maxlen=2)
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = compute_log_loss(y, tx, w)
        grad = compute_logistic_gradient(y, tx, w)
        w -= gamma*grad 

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[0] - losses[1]) < threshold:
            break

    # Final loss
    loss = compute_log_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""
    
     # init parameters
    threshold = 1e-8
    losses = collections.deque(maxlen=2)
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = compute_log_loss(y, tx, w) + lambda_ * lambda_*np.linalg.norm(w)**2
        gradient = compute_logistic_gradient(y, tx, w) + 2*lambda_*w
        w -= gamma*gradient

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[0] - losses[1]) < threshold:
            break


    # Final loss 
    loss = compute_log_loss(y, tx, w)

    return w, loss
    
#####################################################
#              Other Tools                          #
#####################################################
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Authors: Taken from the course homework 03
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

    