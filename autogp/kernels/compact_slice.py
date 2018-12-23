"""
Epanechnikov kernel (order == 2)

There are several options for compact kernel forms (manual selection below):
 - spherical, generic compact, Epanechnikov (current setting), quartic

TODO: add options to __init__

"""

import numpy as np
import tensorflow as tf

from .. import util
from . import kernel
from .kernel_extras import *

class CompactSlice(kernel.Kernel):
    MAX_DIST = 1e8
    MAX_KERN = 1.0
    MIN_KERN = 1e-5

    def __init__(self, input_dim, active_dims=None, lengthscale=1.0,
                 white=0.01, input_scaling=False):
        if input_scaling:
            self.lengthscale = tf.Variable(lengthscale * tf.ones([input_dim]), dtype=tf.float32)
        else:
            self.lengthscale = tf.Variable([lengthscale], dtype=tf.float32)

        self.input_dim = input_dim
        self.active_dims = active_dims
        self.white = white

    def kernel(self, points1, points2=None):
        if points2 is None:
            points2 = points1
            white_noise = (self.white * util.eye(tf.shape(points1)[0]) +
                0.001 * self.white * tf.ones( [tf.shape(points1)[0], tf.shape(points1)[0]]))
        else:
            white_noise = 0.001 * tf.ones( [tf.shape(points1)[0], tf.shape(points2)[0]] )

        points1, points2 = dim_slice(self, points1, points2)

        points1 = points1 / tf.exp(self.lengthscale)
        points2 = points2 / tf.exp(self.lengthscale)
        magnitude_square1 = tf.expand_dims(tf.reduce_sum(points1 ** 2, 1), 1)
        magnitude_square2 = tf.expand_dims(tf.reduce_sum(points2 ** 2, 1), 1)
        # distances = norm(x1-x2)**2 / lengthscale**2
        distances = (magnitude_square1 - 2 * tf.matmul(points1, tf.transpose(points2)) +
                     tf.transpose(magnitude_square2))
        distances = tf.clip_by_value(distances, 0.0, self.MAX_DIST);

        # spherical
        #kern = tf.clip_by_value(1 - (3/2)*tf.sqrt(distances) + 0.5 * tf.sqrt(distances)**3, self.MIN_KERN, self.MAX_KERN)

        # generic compact
        # power = 2 #must be greater than (d+1)/2
        #kern = tf.clip_by_value( (1-tf.sqrt(distances))**power, self.MIN_KERN, self.MAX_KERN )

        # epanechnikov (to a scaling constant)
        power = 2
        kern = (tf.clip_by_value( (1-distances), self.MIN_KERN, self.MAX_KERN ))**power

        # quartic (to a scaling constant)
        #kern = tf.clip_by_value( (1-distances)**2, self.MIN_KERN, self.MAX_KERN )

        return  kern + white_noise

    def diag_kernel(self, points):
        return tf.ones([tf.shape(points)[0]]) # + self.white * tf.random_uniform([tf.shape(points)[0]])

    def get_params(self):
        return [self.lengthscale]

