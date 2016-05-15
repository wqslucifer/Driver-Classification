# -*- coding: utf-8 -*-
"""
Created on 2016/05/14

Modified from https://github.com/felixlaumon/nolearn_utils
For python3 and cv2

@author: qiushi
"""

from __future__ import division
from __future__ import print_function

import sys
import os

import queue
import random
import time
import cv2

import numpy as np
from numpy.random import choice


# image for cv2 imread --> (row,col,chl) 480*640*3

class BaseBatchIterator(object):
    def __init__(self, batch_size, shuffle=False, verbose=False):
        self.batch_size = batch_size
        self.verbose = False
        self.shuffle = shuffle

    def __call__(self, X, y=None):
        self.X, self.y = X, y
        return self

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        n_batches = (n_samples + bs - 1) // bs

        if self.shuffle:
            idx = np.random.permutation(len(self.X))
        else:
            idx = range(len(self.X))

        for i in range(n_batches):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[idx[sl]]
            if self.y is not None:
                yb = self.y[idx[sl]]
            else:
                yb = None
            yield self.transform(Xb, yb)

    @property
    def n_samples(self):
        X = self.X
        if isinstance(X, dict):
            return len(list(X.values())[0])
        else:
            return len(X)

    def transform(self, Xb, yb):
        return Xb, yb

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('X', 'y',):
            if attr in state:
                del state[attr]
        return state


def make_iterator(name, mixin):
    """
    Return an Iterator class added with the provided mixin
    """
    mixin = [BaseBatchIterator] + mixin
    # Reverse the order for type()
    mixin.reverse()
    return type(name, tuple(mixin), {})


class ShuffleBatchIteratorMixin(object):
    """
    From https://github.com/dnouri/nolearn/issues/27#issuecomment-71175381
    Shuffle the order of samples
    """

    def __iter__(self):
        orig_X, orig_y = self.X, self.y
        self.X, self.y = shuffle(self.X, self.y)
        for res in super(ShuffleBatchIteratorMixin, self).__iter__():
            yield res
        self.X = orig_X
        self.y = orig_y


class RandomFlipBatchIteratorMixin(object):
    """
    Randomly flip the random horizontally or vertically
    """

    def __init__(self, flip_horizontal_p=0.5, flip_vertical_p=0.5, *args, **kwargs):
        super(RandomFlipBatchIteratorMixin, self).__init__(*args, **kwargs)
        self.flip_horizontal_p = flip_horizontal_p
        self.flip_vertical_p = flip_vertical_p

    def transform(self, Xb, yb):
        Xb, yb = super(RandomFlipBatchIteratorMixin, self).transform(Xb, yb)
        Xb_flipped = Xb.copy()

        if self.flip_horizontal_p > 0:
            horizontal_flip_idx = get_random_idx(Xb, self.flip_horizontal_p)
            Xb_flipped[horizontal_flip_idx] = Xb_flipped[horizontal_flip_idx][:, ::-1, :]

        if self.flip_vertical_p > 0:
            vertical_flip_idx = get_random_idx(Xb, self.flip_vertical_p)
            Xb_flipped[vertical_flip_idx] = Xb_flipped[vertical_flip_idx][::-1, :, :]

        return Xb_flipped, yb


class RandomCropBatchIteratorMixin(object):
    """
    Randomly crop the image to the desired size
    """

    def __init__(self, crop_size, *args, **kwargs):
        super(RandomCropBatchIteratorMixin, self).__init__(*args, **kwargs)
        self.crop_size = crop_size

    def transform(self, Xb, yb):
        Xb, yb = super(RandomCropBatchIteratorMixin, self).transform(Xb, yb)
        batch_size = min(self.batch_size, Xb.shape[0])
        img_row = Xb.shape[2]
        img_col = Xb.shape[3]
        Xb_transformed = np.empty((batch_size, Xb.shape[1],
                                   self.crop_size[0], self.crop_size[1]), dtype=np.float32)
        for i in range(batch_size):
            start_0 = np.random.choice(img_row - self.crop_size[0])
            end_0 = start_0 + self.crop_size[0]
            start_1 = np.random.choice(img_col - self.crop_size[1])
            end_1 = start_1 + self.crop_size[1]
            Xb_transformed[i] = Xb[i][start_0:end_0, start_1:end_1, :]
        return Xb_transformed, yb


class MeanSubtractBatchiteratorMixin(object):
    """
    Subtract training examples by the given mean
    """

    def __init__(self, mean=None, *args, **kwargs):
        super(MeanSubtractBatchiteratorMixin, self).__init__(*args, **kwargs)
        self.mean = mean

    def transform(self, Xb, yb):
        Xb, yb = super(MeanSubtractBatchiteratorMixin, self).transform(Xb, yb)
        Xb = Xb - self.mean
        return Xb, yb


class LCNBatchIteratorMixin(object):
    """
    Apply local contrast normalization to images
    """

    def __init__(self, lcn_selem=disk(5), *args, **kwargs):
        super(LCNBatchIteratorMixin, self).__init__(*args, **kwargs)
        self.lcn_selem = lcn_selem

    def transform(self, Xb, yb):
        Xb, yb = super(LCNBatchIteratorMixin, self).transform(Xb, yb)
        Xb_transformed = np.asarray([local_contrast_normalization(x, selem=self.lcn_selem) for x in Xb])
        return Xb_transformed, yb


def get_random_idx(arr, p):
    n = arr.shape[0]
    idx = choice(n, int(n * p), replace=False)
    return idx


def shuffle(*arrays):  # arrays need to be >1, arrays is np.array
    p = np.random.permutation(len(arrays[0]))
    return [array[p] for array in arrays]


def disk(radius, dtype=np.uint8):
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)


def local_contrast_normalization(im, selem=disk(5)):
    img = np.zeros(im.shape)
    img = (im * 255).astype(np.uint8)
    if len(img.shape) <= 2:  # gray
        img = cv2.equalizeHist(img, selem)
    else:  # color
        for i in range(3):
            img[:, :, i] = cv2.equalizeHist(img[:, :, i], selem)
    img = img.astype(np.float32) / 255
    return img
