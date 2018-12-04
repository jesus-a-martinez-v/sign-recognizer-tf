import math

import h5py
import numpy as np


def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', 'r')
    X_train = np.array(train_dataset['train_set_x'][:])
    y_train = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File('datasets/test_signs.h5', 'r')
    X_test = np.array(test_dataset['test_set_x'][:])
    y_test = np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset['list_classes'][:])

    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))

    return X_train, y_train, X_test, y_test, classes


def one_hot_encode(y, number_of_classes):
    return np.eye(number_of_classes)[y.reshape(-1)].T


def random_mini_batches(X, y, mini_batch_size=64, seed=0):
    number_of_training_examples = X.shape[0]
    mini_batches = []

    np.random.seed(seed)

    # Shuffle
    permutation = list(np.random.permutation(number_of_training_examples))
    shuffled_X = X[permutation, :, :, :]
    shuffled_y = y[permutation, :]

    # Partition
    number_of_complete_mini_batches = math.floor(number_of_training_examples / mini_batch_size)

    for k in range(number_of_complete_mini_batches):
        start = k * mini_batch_size
        end = start + mini_batch_size
        mini_batch_X = shuffled_X[start:end, :, :, :]
        mini_batch_y = shuffled_y[start:end, :]

        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if number_of_training_examples % mini_batch_size != 0:
        mini_batch_X = shuffled_X[number_of_complete_mini_batches * mini_batch_size:number_of_training_examples, :, :,
                       :]
        mini_batch_y = shuffled_y[number_of_complete_mini_batches * mini_batch_size:number_of_training_examples, :]

        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)

    return mini_batches
