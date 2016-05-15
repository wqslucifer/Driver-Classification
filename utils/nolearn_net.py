# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:24:10 2016

@author: qiushi
"""

import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import dnn
from nolearn.lasagne import objective
from nolearn.lasagne.handlers import SaveWeights
from nolearn.lasagne import NeuralNet as BaseNeuralNet


class NeuralNet(BaseNeuralNet):
    def transform(self, X, target_layer_name, y=None):
        target_layer = self.layers_[target_layer_name]
        layers = self.layers_
        input_layers = [
            layer for layer in layers.values()
            if isinstance(layer, lasagne.layers.InputLayer)
            ]
        X_inputs = [
            theano.Param(input_layer.input_var, name=input_layer.name)
            for input_layer in input_layers
            ]
        target_layer_output = lasagne.layers.get_output(
                target_layer, None, deterministic=True
        )
        transform_iter = theano.function(
                inputs=X_inputs,
                outputs=target_layer_output,
                allow_input_downcast=True,
        )
        outputs = []
        for Xb, yb in self.batch_iterator_test(X):
            outputs.append(self.apply_batch_func(transform_iter, Xb))
        return np.vstack(outputs)
