import numpy as np
import math


def random_mini_batches_standard(X, Y, mini_batch_size, seed):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def random_mini_batches(X1, X2, X3, X4, X5, X1_FULL, X2_FULL, X3_FULL, X4_FULL, X5_FULL, Y, mini_batch_size, seed):
    m = X1.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X1 = X1[permutation, :]
    shuffled_X2 = X2[permutation, :]
    shuffled_X3 = X3[permutation, :]
    shuffled_X4 = X4[permutation, :]
    shuffled_X5 = X5[permutation, :]
    shuffled_X1_FULL = X1_FULL[permutation, :]
    shuffled_X2_FULL = X2_FULL[permutation, :]
    shuffled_X3_FULL = X3_FULL[permutation, :]
    shuffled_X4_FULL = X4_FULL[permutation, :]
    shuffled_X5_FULL = X5_FULL[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X1 = shuffled_X1[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X2 = shuffled_X2[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X3 = shuffled_X3[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X4 = shuffled_X4[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X5 = shuffled_X5[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X1_FULL = shuffled_X1_FULL[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X2_FULL = shuffled_X2_FULL[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X3_FULL = shuffled_X3_FULL[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X4_FULL = shuffled_X4_FULL[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X5_FULL = shuffled_X5_FULL[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X1, mini_batch_X2, mini_batch_X3, mini_batch_X4, mini_batch_X5,
                      mini_batch_X1_FULL, mini_batch_X2_FULL, mini_batch_X3_FULL, mini_batch_X4_FULL, mini_batch_X5_FULL, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X1 = shuffled_X1[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X2 = shuffled_X2[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X3 = shuffled_X3[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X4 = shuffled_X4[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X5 = shuffled_X5[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X1_FULL = shuffled_X1_FULL[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X2_FULL = shuffled_X2_FULL[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X3_FULL = shuffled_X3_FULL[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X4_FULL = shuffled_X4_FULL[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X5_FULL = shuffled_X5_FULL[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X1, mini_batch_X2, mini_batch_X3, mini_batch_X4, mini_batch_X5,
                      mini_batch_X1_FULL, mini_batch_X2_FULL, mini_batch_X3_FULL, mini_batch_X4_FULL, mini_batch_X5_FULL, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def random_mini_batches_standardtwoModality(X1, X2, Y, mini_batch_size, seed):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X1.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X1 = X1[permutation, :]
    shuffled_X2 = X2[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X1 = shuffled_X1[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_X2 = shuffled_X2[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X1, mini_batch_X2, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X1 = shuffled_X1[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_X2 = shuffled_X2[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X1, mini_batch_X2, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
