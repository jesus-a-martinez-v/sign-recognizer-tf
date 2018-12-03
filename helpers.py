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