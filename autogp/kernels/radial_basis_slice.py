'''
Modification to radial_basis.py to allow kernel to operate on subset of features
'''

import numpy as np
import tensorflow as tf

from .. import util
from . import kernel
from .kernel_extras import *

class RadialBasisSlice(kernel.Kernel):
    MAX_DIST = 1e7

    def __init__(self, input_dim, active_dims=None, lengthscale=1.0, std_dev=1.0,
                 white=0.01, input_scaling=False):
        if input_scaling:
            self.lengthscale = tf.Variable(lengthscale * tf.ones([input_dim]), dtype=tf.float32)
        else:
            self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32)

        self.std_dev = tf.Variable([std_dev], dtype=tf.float32)
        self.input_dim = input_dim
        self.active_dims = active_dims
        self.white = white

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = (self.white * util.eye(tf.shape(points1)[0]) +
                0.1 * self.white * tf.ones( [tf.shape(points1)[0], tf.shape(points1)[0]]))
        else:
            white_noise = 0.001 * tf.ones( [tf.shape(points1)[0], tf.shape(points2)[0]] )

        points1, points2 = dim_slice(self, points1, points2)

        points1 = points1 / tf.exp(self.lengthscale)
        points2 = points2 / tf.exp(self.lengthscale)
        magnitude_square1 = tf.expand_dims(tf.reduce_sum(points1 ** 2, 1), 1)
        magnitude_square2 = tf.expand_dims(tf.reduce_sum(points2 ** 2, 1), 1)
        distances = (magnitude_square1 - 2 * tf.matmul(points1, tf.transpose(points2)) +
                     tf.transpose(magnitude_square2))
        distances = tf.clip_by_value(distances, 0.0, self.MAX_DIST);

        kern = (tf.exp(self.std_dev) * tf.exp(-distances / 2.0))
        return kern + white_noise

    def diag_kernel(self, points):
        return (tf.exp(self.std_dev) + self.white) * tf.ones([tf.shape(points)[0]]) # + self.white * tf.random_uniform([tf.shape(points)[0]])

    def get_params(self):
        return [self.lengthscale, self.std_dev]

