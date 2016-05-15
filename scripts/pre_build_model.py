# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:24:10 2016

@author: qiushi
"""

import os
import time

import lasagne
import numpy as np
import theano
from lasagne import layers
from lasagne.layers.dnn import Conv2DDNNLayer as conv2dbn
from nolearn.lasagne import PrintLayerInfo
from nolearn.lasagne.handlers import SaveWeights

from utils.hooks import EarlyStopping, StepDecay
from utils.iterators import (
    ShuffleBatchIteratorMixin,
    make_iterator
)
from utils.layers import batch_norm
from utils.nolearn_net import NeuralNet

start_time = time.time()

img_cols = 4 * 16
img_rows = 3 * 16
n_classes = 10
batch_size = 128

main_path1 = 'F:\\kaggle\\Driver Classification'
main_path2 = 'F:\\kaggle\\State Farm Distracted Driver Detection\\'
model_filename = os.path.join(main_path1, 'model/vgg16.pkl')

layer_info = PrintLayerInfo()


def conv2dbn(l, name, **kwargs):
    l = layers.dnn.Conv2DDNNLayer(
            l, name=name,
            **kwargs
    )
    l = batch_norm(l, name='%sbn' % name)
    return l


conv_kwargs = dict(
        pad='same',
        flip_filters=False,  # added
        nonlinearity=lasagne.nonlinearities.very_leaky_rectify
)

train_iterator_mixins = [
    ShuffleBatchIteratorMixin,
    make_iterator
]

train_iterator_kwargs = dict(
        batch_size=batch_size,
)

# iterators
TrainIterator = make_iterator('TrainIterator', train_iterator_mixins)
save_weights = SaveWeights(model_filename, only_best=True, pickle=False)
train_iterator = TrainIterator(**train_iterator_kwargs)

# build vgg16
l = layers.InputLayer(name='input', shape=(None, 3, img_rows, img_cols))

# conv1
l = conv2dbn(l, name='conv1_1', num_filters=64, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='conv1_2', num_filters=64, filter_size=(3, 3), **conv_kwargs)
l = layers.dnn.Pool2DDNNLayer(l, name='pool1', pool_size=2, mode='max')

# conv2
l = conv2dbn(l, name='conv2_1', num_filters=128, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='conv2_2', num_filters=128, filter_size=(3, 3), **conv_kwargs)
l = layers.dnn.Pool2DDNNLayer(l, name='pool2', pool_size=2, mode='max')

# conv3
l = conv2dbn(l, name='conv3_1', num_filters=256, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='conv3_2', num_filters=256, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='conv3_3', num_filters=256, filter_size=(3, 3), **conv_kwargs)
l = layers.dnn.Pool2DDNNLayer(l, name='pool3', pool_size=2, mode='max')

# conv4
l = conv2dbn(l, name='conv4_1', num_filters=256, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='conv4_2', num_filters=256, filter_size=(3, 3), **conv_kwargs)
l = conv2dbn(l, name='conv4_3', num_filters=256, filter_size=(3, 3), **conv_kwargs)
l = layers.dnn.Pool2DDNNLayer(l, name='pool4', pool_size=2, mode='max')

# full connect
l = layers.DenseLayer(l, name='dens1', num_units=4096)
l = layers.DropoutLayer(l, name='drop1', p=0.5)
l = layers.DenseLayer(l, name='dens2', num_units=4096)
l = layers.DropoutLayer(l, name='drop2', p=0.5)

l = layers.DenseLayer(l, name='dens3', num_units=1000, nonlinearity=None)
l = layers.DenseLayer(l, name='out', num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)

net = NeuralNet(
        layers=l,

        regression=False,  # flag to indicate if we're dealing with regression problem
        use_label_encoder=False,

        objective_l2=1e-6,

        update=lasagne.updates.nesterov_momentum,
        update_learning_rate=theano.shared(np.float32(0.03)),

        # train_split=TrainSplit(0.15, random_state=42, stratify=False),
        batch_iterator_train=train_iterator,  # Data augmentation
        # batch_iterator_test=test_iterator,

        on_epoch_finished=[
            StepDecay('update_learning_rate', start=0.03, stop=1e-5),  # update learning rate
            EarlyStopping(patience=50),
            save_weights,
            # save_training_history,
            # plot_training_history,
        ],
        verbose=10,  # print out information during training by setting verbose=1

        max_epochs=50,
)

net.initialize()
layer_info(net)

'''
name       size         total    cap.Y    cap.X    cov.Y    cov.X    filter Y    filter X    field Y    field X
---------  ---------  -------  -------  -------  -------  -------  ----------  ----------  ---------  ---------
input      3x48x64       9216   100.00   100.00   100.00   100.00          48          64         48         64
conv1_1    64x48x64    196608   100.00   100.00     6.25     4.69           3           3          3          3
conv1_1bn  64x48x64    196608   100.00   100.00   100.00   100.00          48          64         48         64
conv1_2    64x48x64    196608   100.00   100.00   100.00   100.00          48          64         48         64
conv1_2bn  64x48x64    196608   100.00   100.00   100.00   100.00          48          64         48         64
pool1      64x24x32     49152   100.00   100.00   100.00   100.00          48          64         48         64
conv2_1    128x24x32    98304   100.00   100.00   100.00   100.00          48          64         48         64
conv2_1bn  128x24x32    98304   100.00   100.00   100.00   100.00          48          64         48         64
conv2_2    128x24x32    98304   100.00   100.00   100.00   100.00          48          64         48         64
conv2_2bn  128x24x32    98304   100.00   100.00   100.00   100.00          48          64         48         64
pool2      128x12x16    24576   100.00   100.00   100.00   100.00          48          64         48         64
conv3_1    256x12x16    49152   100.00   100.00   100.00   100.00          48          64         48         64
conv3_1bn  256x12x16    49152   100.00   100.00   100.00   100.00          48          64         48         64
conv3_2    256x12x16    49152   100.00   100.00   100.00   100.00          48          64         48         64
conv3_2bn  256x12x16    49152   100.00   100.00   100.00   100.00          48          64         48         64
conv3_3    256x12x16    49152   100.00   100.00   100.00   100.00          48          64         48         64
conv3_3bn  256x12x16    49152   100.00   100.00   100.00   100.00          48          64         48         64
pool3      256x6x8      12288   100.00   100.00   100.00   100.00          48          64         48         64
conv4_1    256x6x8      12288   100.00   100.00   100.00   100.00          48          64         48         64
conv4_1bn  256x6x8      12288   100.00   100.00   100.00   100.00          48          64         48         64
conv4_2    256x6x8      12288   100.00   100.00   100.00   100.00          48          64         48         64
conv4_2bn  256x6x8      12288   100.00   100.00   100.00   100.00          48          64         48         64
conv4_3    256x6x8      12288   100.00   100.00   100.00   100.00          48          64         48         64
conv4_3bn  256x6x8      12288   100.00   100.00   100.00   100.00          48          64         48         64
pool4      256x3x4       3072   100.00   100.00   100.00   100.00          48          64         48         64
dens1      4096          4096   100.00   100.00   100.00   100.00          48          64         48         64
drop1      4096          4096   100.00   100.00   100.00   100.00          48          64         48         64
dens2      4096          4096   100.00   100.00   100.00   100.00          48          64         48         64
drop2      4096          4096   100.00   100.00   100.00   100.00          48          64         48         64
dens3      1000          1000   100.00   100.00   100.00   100.00          48          64         48         64
out        1000          1000   100.00   100.00   100.00   100.00          48          64         48         64

Explanation
    X, Y:    image dimensions
    cap.:    learning capacity
    cov.:    coverage of image
    magenta: capacity too low (<1/6)
    cyan:    image coverage too high (>100%)
    red:     capacity too low and coverage too high
'''
# net2.fit(X, y)


end_time = time.time()

running_time = end_time - start_time

print('running time: %d days, %dh:%dm:%ds.' % (divmod(running_time, 86400)[0], divmod(running_time, 3600)[0],
                                               divmod(running_time, 60)[0], divmod(running_time, 60)[1]))
