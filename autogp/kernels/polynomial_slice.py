'''
Polynomial kernel
Code adapted from GPflow for autogp
'''

import numpy as np
import tensorflow as tf

from .. import util
from . import kernel
from .kernel_extras import *


class PolynomialSlice(kernel.Kernel):
    # TODO implement matrix power
    def __init__(self, input_dim, active_dims=None, order=2, std_dev=1.0,
                 white=0.01, bias=0.01, input_scaling=False):
        if input_scaling:
            self.std_dev = tf.Variable(std_dev * tf.ones([input_dim]))
        else:
            self.std_dev = tf.Variable([std_dev], dtype=tf.float32)
        if bias:
            self.bias = tf.Variable([bias], dtype=tf.float32)
        self.order = order
        self.input_dim = input_dim
        self.active_dims = active_dims
        self.white = white

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = (self.white * util.eye(tf.shape(points1)[0]) +
                0.1 * self.white * tf.random_uniform( [tf.shape(points1)[0], tf.shape(points1)[0]], minval = 0.5 ))
        else:
            white_noise = 0.1 * self.white * tf.random_uniform( [tf.shape(points1)[0], tf.shape(points2)[0]], minval = 0.5 )

        points1, points2 = dim_slice(self, points1, points2)
        points1 = tf.exp(self.std_dev)*points1
        kern = tf.matmul(points1, points2, transpose_b=True) + tf.exp(self.bias)
        return kern ** self.order + white_noise


    def diag_kernel(self, points):
        points = dim_slice_diag(self, points)
        return (tf.reduce_sum(tf.square(points) * tf.exp(self.std_dev), 1) + tf.exp(self.bias)) ** self.order + self.white

    def get_params(self):
        return [self.std_dev]
