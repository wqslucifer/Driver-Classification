# -*- coding: utf-8 -*-
"""
Created on 2016/05/14

Modified from https://github.com/felixlaumon/nolearn_utils
For python3

@author: qiushi
"""

import numpy as np


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, lasagne, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, lasagne.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(lasagne, self.name).set_value(new_value)


class EarlyStopping(object):
    """From https://github.com/dnouri/kfkd-tutorial"""

    def __init__(self, patience=50):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, lasagne, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_train = train_history[-1]['train_loss']
        current_epoch = train_history[-1]['epoch']

        # Ignore if training loss is greater than valid loss
        if current_train > current_valid:
            return

        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in lasagne.get_all_params()]
        elif self.best_valid_epoch + self.patience <= current_epoch:
            print('Early stopping.')
            print('Best valid loss was {:.6f} at epoch {}.'.format(
                    self.best_valid, self.best_valid_epoch))
            lasagne.load_weights_from(self.best_weights)
            raise StopIteration()


class StepDecay(object):
    """From https://github.com/dnouri/kfkd-tutorial"""
    '''
    Step decay: Reduce the learning rate by some factor
    every few epochs.
    Typical values might be reducing the learning rate
    by a half every 5 epochs, or by 0.1 every 20 epochs.
    These numbers depend heavily on the type of problem
    and the model.
    One heuristic you may see in practice is to watch
    the validation error while training with a fixed
    learning rate, and reduce the learning rate by a
    constant (e.g. 0.5) whenever the validation error
    stops improving.
    '''

    def __init__(self, name, start=0.03, stop=0.001, delay=0):
        self.name = name
        self.delay = delay
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, lasagne, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop,
                                  lasagne.max_epochs - self.delay)

        epoch = train_history[-1]['epoch'] - self.delay
        if epoch >= 0:
            new_value = np.float32(self.ls[epoch - 1])
            getattr(lasagne, self.name).set_value(new_value)
