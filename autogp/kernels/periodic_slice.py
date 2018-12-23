
"""
The periodic kernel. Defined in  Equation (47) of Mackay (Intro to GPs)
Code adapted from GPflow
"""

import numpy as np
import tensorflow as tf

from .. import util
from . import kernel
from .kernel_extras import *


class PeriodicSlice(kernel.Kernel):
    MAX_DIST = 1e7

    def __init__(self, input_dim, active_dims=None, period=1.0, lengthscale=1.0, std_dev=1.0,
                 white=0.01):
        # ARD (input scaling) not currently supported
        self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32)
        self.std_dev = tf.Variable([std_dev], dtype=tf.float32)
        self.period = tf.Variable([period], dtype=tf.float32)
        self.input_dim = input_dim
        self.active_dims = active_dims
        self.white = white

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = (self.white * util.eye(tf.shape(points1)[0]) +
                0.01 * self.white * tf.ones( [tf.shape(points1)[0], tf.shape(points1)[0]] ))
        else:
            white_noise = 0.0001 * tf.ones( [tf.shape(points1)[0], tf.shape(points2)[0]] )

        points1, points2 = dim_slice(self, points1, points2)

        r = tf.sin((np.pi/self.period) * (points1-tf.transpose(points2)))
        r = tf.square(r / tf.exp(self.lengthscale))
        kern = (tf.exp(self.std_dev) * tf.exp(-r / 2.0))
        return kern + white_noise

    def diag_kernel(self, points):
        return tf.exp(self.std_dev) * tf.ones([tf.shape(points)[0]]) + self.white * tf.ones([tf.shape(points)[0]])

    def get_params(self):
        return [self.lengthscale, self.period, self.std_dev]
